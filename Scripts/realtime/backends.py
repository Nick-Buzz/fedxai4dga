"""Inference backends: every way the trained models can be served.

Each backend wraps one trained model behind ``predict(X) -> probabilities``,
where ``X`` is a float32 NumPy array of scaled features and the result is a
float32 NumPy array of P(DGA), returned to host memory.  Returning to host is
part of the timed work, because a verdict is only useful once the caller has it.

Keras models are served four ways, from the most naive to the most optimised:

  keras-predict   ``model.predict`` -- the path the training scripts use; it
                  builds a tf.data pipeline per call, so it carries a large
                  fixed overhead that dominates small batches
  tf-eager        ``model(x)`` op by op, no graph
  tf-function     one traced graph with a dynamic batch dimension
  tf-xla          the same graph compiled with XLA (one compilation per shape)
  onnxruntime     the graph exported to ONNX and run by ONNX Runtime (CPU),
                  the usual production serving path for small models

XGBoost is served with ``inplace_predict``, which skips DMatrix construction,
on CPU (NumPy input) or on CUDA (CuPy input).

Serving path per device, used by the pipeline, full-test and stream suites,
chosen from the measured batch-1 latency: ONNX Runtime on CPU, XLA on GPU.

The device is fixed per process by the caller (``CUDA_VISIBLE_DEVICES``) and the
CPU budget by thread settings plus CPU affinity, so a backend never has to
choose.  See ``Scripts/realtime/worker.py``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERAS_MODELS = ("exat_mlp", "no_attention", "single_head", "plain_mlp")
ALL_MODELS = KERAS_MODELS + ("xgboost",)
KERAS_BACKENDS = ("keras-predict", "tf-eager", "tf-function", "tf-xla", "onnxruntime")
MODEL_LABELS = {
    "exat_mlp": "ExAt-MLP",
    "no_attention": "ExAt-MLP w/o attention",
    "single_head": "ExAt-MLP, 1 head",
    "plain_mlp": "MLP",
    "xgboost": "XGBoost",
}


@dataclass
class Backend:
    model: str
    backend: str
    device: str
    threads: int
    predict: Callable[[np.ndarray], np.ndarray]
    load_seconds: float
    info: dict = field(default_factory=dict)


def model_dir(results_dir: Path, model: str, seed: int) -> Path:
    return Path(results_dir) / model / f"seed_{seed}"


def onnx_path(output_root: Path, model: str, seed: int) -> Path:
    return Path(output_root) / "onnx" / f"{model}_seed{seed}.onnx"


# --------------------------------------------------------------------- keras

_KERAS_CACHE: dict = {}


def _load_keras(path: Path):
    import tensorflow as tf
    import Models.exat_mlp  # noqa: F401  registers FeatureTokenizer for deserialisation

    key = str(path)
    if key not in _KERAS_CACHE:
        _KERAS_CACHE[key] = tf.keras.models.load_model(path, compile=False)
    return _KERAS_CACHE[key]


def keras_backend(model: str, backend: str, device: str, threads: int,
                  results_dir: Path, seed: int) -> Backend:
    import tensorflow as tf

    path = model_dir(results_dir, model, seed) / "model.keras"
    t0 = time.perf_counter()
    net = _load_keras(path)
    n_features = int(net.inputs[0].shape[-1])
    spec = [tf.TensorSpec([None, n_features], tf.float32, name="features")]

    if backend == "keras-predict":
        def predict(X):
            return net.predict(X, batch_size=len(X), verbose=0).reshape(-1)
    elif backend == "tf-eager":
        def predict(X):
            return net(tf.convert_to_tensor(X), training=False).numpy().reshape(-1)
    elif backend in ("tf-function", "tf-xla"):
        graph = tf.function(lambda x: net(x, training=False), input_signature=spec,
                            jit_compile=(backend == "tf-xla"), reduce_retracing=True)

        def predict(X):
            return graph(tf.convert_to_tensor(X)).numpy().reshape(-1)
    else:
        raise ValueError(f"unknown keras backend {backend!r}")

    return Backend(model, backend, device, threads, predict, time.perf_counter() - t0, {
        "parameters": int(net.count_params()),
        "artifact_bytes": path.stat().st_size,
        "n_features": n_features,
    })


def export_onnx(model: str, results_dir: Path, seed: int, output_root: Path) -> Path:
    """Convert a saved Keras model to ONNX once; later calls reuse the file."""
    out = onnx_path(output_root, model, seed)
    if out.exists():
        return out
    import tensorflow as tf
    import tf2onnx

    net = _load_keras(model_dir(results_dir, model, seed) / "model.keras")
    spec = (tf.TensorSpec([None, int(net.inputs[0].shape[-1])], tf.float32, name="features"),)
    out.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(net, input_signature=spec, opset=17, output_path=str(out))
    return out


def onnx_backend(model: str, threads: int, results_dir: Path, seed: int,
                 output_root: Path) -> Backend:
    import onnxruntime as ort

    path = onnx_path(output_root, model, seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the orchestrator's export step first")
    t0 = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def predict(X):
        return session.run(None, {input_name: X})[0].reshape(-1)

    return Backend(model, "onnxruntime", "cpu", threads, predict, time.perf_counter() - t0, {
        "artifact_bytes": path.stat().st_size,
        "n_features": int(session.get_inputs()[0].shape[-1]),
    })


# ------------------------------------------------------------------- xgboost

def _import_cupy():
    """Import CuPy, preloading NVRTC from the pip CUDA wheels TensorFlow installed."""
    import ctypes
    import glob
    try:
        import nvidia.cuda_nvrtc as nvrtc
        for lib in sorted(glob.glob(os.path.join(list(nvrtc.__path__)[0], "lib", "libnvrtc*.so*"))):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except ImportError:
        pass
    import cupy
    return cupy


def xgboost_backend(device: str, threads: int, results_dir: Path, seed: int) -> Backend:
    import xgboost as xgb

    path = model_dir(results_dir, "xgboost", seed) / "model.json"
    t0 = time.perf_counter()
    booster = xgb.Booster()
    booster.load_model(str(path))
    booster.set_param({"device": "cuda" if device == "gpu" else "cpu", "nthread": threads})

    if device == "gpu":
        # A CUDA booster given a NumPy array silently falls back to CPU
        # prediction through a DMatrix, so the input must be on the device.
        # The host->device copy and the copy back are part of the timed call.
        cp = _import_cupy()

        def predict(X):
            return cp.asnumpy(booster.inplace_predict(cp.asarray(X))).reshape(-1)
    else:
        def predict(X):
            return booster.inplace_predict(X).reshape(-1)

    return Backend("xgboost", "xgb-inplace", device, threads, predict, time.perf_counter() - t0, {
        "trees": int(booster.num_boosted_rounds()),
        "artifact_bytes": path.stat().st_size,
        "n_features": int(booster.num_features()),
    })


# ------------------------------------------------------------------ dispatch

def make_backend(model: str, backend: str, device: str, threads: int,
                 results_dir: Path, seed: int, output_root: Path) -> Backend:
    if model == "xgboost":
        return xgboost_backend(device, threads, results_dir, seed)
    if backend == "onnxruntime":
        return onnx_backend(model, threads, results_dir, seed, output_root)
    return keras_backend(model, backend, device, threads, results_dir, seed)


def backends_for(model: str, device: str) -> list[str]:
    """The backends that make sense for a model on a device."""
    if model == "xgboost":
        return ["xgb-inplace"]
    if device == "gpu":
        return ["keras-predict", "tf-eager", "tf-function", "tf-xla"]
    return list(KERAS_BACKENDS)


def configure_threads(threads: int, slot: int = 0) -> list[int]:
    """Limit this process to ``threads`` logical CPUs, one per physical core first.

    Thread-count settings alone do not bound CPU use (XGBoost, oneDNN and ORT
    each keep their own pools), so the process is also pinned with
    ``sched_setaffinity``, which is what a container CPU quota approximates.
    Even-numbered logical CPUs are taken first so that, on an SMT machine,
    low thread counts land on distinct physical cores.  ``slot`` selects the
    next disjoint group, so parallel shards do not share CPUs until the
    machine runs out of them.
    """
    available = sorted(os.sched_getaffinity(0))
    order = available[0::2] + available[1::2]
    chosen = sorted({order[(slot * threads + k) % len(order)] for k in range(threads)})
    os.sched_setaffinity(0, chosen)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS"):
        os.environ[var] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1" if threads == 1 else "2"
    return chosen


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Export the Keras models to ONNX (idempotent).")
    ap.add_argument("--models", nargs="+", default=list(KERAS_MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "Results/full_run")
    ap.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Results/realtime")
    cli = ap.parse_args()
    for name in cli.models:
        print(f"  onnx {export_onnx(name, cli.results_dir, cli.seed, cli.output_root)}", flush=True)
