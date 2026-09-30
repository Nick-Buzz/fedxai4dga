"""One benchmark process: a fixed device and a fixed CPU budget.

Thread pools are sized when TensorFlow, ONNX Runtime and XGBoost first start,
so every (device, threads) point runs in a fresh process launched by
``Scripts/realtime/run_benchmark.py``.  Suites, each written to its own CSV:

  latency    model-only latency and throughput across batch sizes and backends
  coldstart  model load time, first-call (tracing) and steady-state latency
  fidelity   each backend's probabilities against the stored test predictions
  fulltest   the whole 294,000-row test set scored through the serving path,
             so the timed model is shown to be the model whose accuracy is reported
  pipeline   raw domain name -> features -> verdict, timed per stage
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The deployable path per device: what the pipeline and full-test suites use.
FULLTEST_BATCH = 1024
SERVING_BACKENDS = {"cpu": ("onnxruntime", "xgb-inplace"), "gpu": ("tf-xla", "xgb-inplace")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--tag", required=True, help="suffix of the CSVs this process writes")
    p.add_argument("--suites", nargs="+", default=["latency"],
                   choices=["latency", "coldstart", "fidelity", "fulltest", "pipeline"])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--backends", nargs="+", default=None, help="restrict to these backends")
    p.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    p.add_argument("--pipeline-batch-sizes", nargs="+", type=int, default=[1, 8, 64, 512, 4096])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "Results/full_run")
    p.add_argument("--test-file", type=Path, default=PROJECT_ROOT / "Data/Processed/test_data.csv")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Results/realtime")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--naive-max-batch", type=int, default=1024,
                   help="largest batch for keras-predict and tf-eager; TensorFlow's CPU "
                        "allocator keeps its peak, and eager attention at 4k rows exhausts 7.6 GB")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=6.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_ROOT))
    from Scripts.realtime.backends import configure_threads

    cpus = configure_threads(args.threads)       # before any numerical library starts

    import time
    import warnings
    t_import = time.perf_counter()
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    tf_import_seconds = time.perf_counter() - t_import
    tf.get_logger().setLevel("ERROR")
    warnings.filterwarnings("ignore")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1 if args.threads == 1 else 2)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.device == "gpu" and not gpus:
        raise SystemExit("GPU worker started but TensorFlow sees no GPU")

    from Scripts.realtime.backends import backends_for, make_backend
    from Scripts.realtime.timing import latency_summary, measure
    from Scripts.realtime.featurizer import MODEL_FEATURES

    header = pd.read_csv(args.test_file, nrows=0).columns
    features = [c for c in header if c not in ("Name", "Label", "Family")]
    if features != MODEL_FEATURES:
        raise SystemExit("test file feature order differs from the featurizer's")
    test = pd.read_csv(args.test_file, dtype={c: np.float32 for c in features})
    X = np.ascontiguousarray(test[features].to_numpy(np.float32))
    y = test["Label"].to_numpy(np.int8)
    names = test["Name"].tolist()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = {"device": args.device, "threads": args.threads, "cpus": " ".join(map(str, cpus))}
    print(f"[{args.tag}] device={args.device} threads={args.threads} cpus={cpus} "
          f"rows={len(X):,}", flush=True)

    rows = {s: [] for s in args.suites}
    featurizer = None
    if "pipeline" in rows:
        from Scripts.realtime.featurizer import DomainFeaturizer
        featurizer = DomainFeaturizer()

    def gpu_peak_mb():
        if args.device != "gpu":
            return None
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20

    for model in args.models:
        stored = None
        if {"fidelity", "fulltest"} & set(args.suites):
            stored = pd.read_csv(args.results_dir / model / f"seed_{args.seed}" / "test_predictions.csv.gz")
            # A run may have scored only the leading rows of the test file
            # (run_experiments.py --smoke-test); compare on those rows.
            n_ref = len(stored)
            if stored["Name"].tolist() != names[:n_ref]:
                raise SystemExit(f"{model}: stored predictions are not in test-file order")
            reference = stored["probability"].to_numpy(np.float32)
            X_ref, y_ref = X[:n_ref], y[:n_ref]

        # Serving paths first: the TensorFlow CPU allocator keeps its peak, so a
        # naive backend's large batch must not run before the paths that matter.
        ordered = sorted(backends_for(model, args.device),
                         key=lambda b: b not in SERVING_BACKENDS[args.device])
        for backend_name in ordered:
            if args.backends and backend_name not in args.backends:
                continue
            try:
                t0 = time.perf_counter()
                backend = make_backend(model, backend_name, args.device, args.threads,
                                       args.results_dir, args.seed, args.output_root)
                load_seconds = time.perf_counter() - t0
                t0 = time.perf_counter()
                backend.predict(X[:1])
                first_call = time.perf_counter() - t0
            except Exception as exc:      # a backend that cannot load is reported, not fatal
                print(f"[{args.tag}] {model}/{backend_name}: FAILED to load: {exc}", flush=True)
                continue
            if args.device == "gpu":
                tf.config.experimental.reset_memory_stats("GPU:0")
            ident = {"model": model, "backend": backend_name, **common}

            if "coldstart" in rows:
                steady = [None] * 50
                for k in range(50):
                    t0 = time.perf_counter()
                    backend.predict(X[k + 1:k + 2])
                    steady[k] = time.perf_counter() - t0
                rows["coldstart"].append({
                    **ident, "tf_import_s": tf_import_seconds, "load_s": load_seconds,
                    "first_call_ms": first_call * 1e3,
                    "steady_median_ms": float(np.median(steady)) * 1e3,
                    **backend.info,
                })

            if "fidelity" in rows:
                n = min(4096, n_ref)
                try:     # in chunks: the allocators keep the peak of earlier backends
                    p = np.concatenate([backend.predict(X[i:i + 512]) for i in range(0, n, 512)])
                except Exception as exc:
                    print(f"[{args.tag}] {model}/{backend_name} fidelity FAILED: "
                          f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
                    p = None
            if "fidelity" in rows and p is not None:
                rows["fidelity"].append({
                    **ident, "rows": n,
                    "max_abs_diff": float(np.abs(p - reference[:n]).max()),
                    "mean_abs_diff": float(np.abs(p - reference[:n]).mean()),
                    "verdict_agreement": float(((p >= 0.5) == (reference[:n] >= 0.5)).mean()),
                })

            if "fulltest" in rows and backend_name in SERVING_BACKENDS[args.device]:
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                import json
                t0 = time.perf_counter()
                try:
                    p = np.concatenate([backend.predict(X_ref[i:i + FULLTEST_BATCH])
                                        for i in range(0, len(X_ref), FULLTEST_BATCH)])
                except Exception as exc:
                    print(f"[{args.tag}] {model}/{backend_name} full test FAILED: "
                          f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
                    p = None
                wall = time.perf_counter() - t0
                reported = json.loads((args.results_dir / model / f"seed_{args.seed}" / "metrics.json").read_text())
            if "fulltest" in rows and backend_name in SERVING_BACKENDS[args.device] and p is not None:
                rows["fulltest"].append({
                    **ident, "rows": n_ref, "batch_size": FULLTEST_BATCH, "seconds": wall,
                    "throughput_per_s": n_ref / wall,
                    "accuracy": accuracy_score(y_ref, p >= 0.5), "f1": f1_score(y_ref, p >= 0.5),
                    "roc_auc": roc_auc_score(y_ref, p),
                    "reported_accuracy": reported["accuracy"], "reported_f1": reported["f1"],
                    "reported_roc_auc": reported["roc_auc"],
                    "max_abs_diff_vs_stored": float(np.abs(p - reference).max()),
                })
                print(f"[{args.tag}] {model}/{backend_name} full test: "
                      f"{n_ref / wall:,.0f}/s acc {rows['fulltest'][-1]['accuracy']:.4f} "
                      f"(reported {reported['accuracy']:.4f})", flush=True)

            if "latency" in rows:
                for bs in args.batch_sizes:
                    if backend_name in ("keras-predict", "tf-eager") and bs > args.naive_max_batch:
                        break
                    try:
                        stats = measure(backend.predict, X, bs, min_seconds=args.min_seconds,
                                        max_seconds=args.max_seconds)
                    except Exception as exc:   # e.g. out of memory at the largest batch
                        print(f"[{args.tag}] {model}/{backend_name} bs={bs}: FAILED: "
                              f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
                        break
                    rows["latency"].append({**ident, "batch_size": bs, **stats,
                                            "gpu_peak_mb": gpu_peak_mb(), **backend.info})
                    print(f"[{args.tag}] {model:12s} {backend_name:13s} bs={bs:6d} "
                          f"p50 {stats['latency_ms_p50']:9.3f} ms  p99 {stats['latency_ms_p99']:9.3f} ms  "
                          f"{stats['throughput_per_s']:>12,.0f}/s  cores {stats['cpu_cores_busy']:.1f}",
                          flush=True)

            if "pipeline" in rows and backend_name in SERVING_BACKENDS[args.device]:
                for bs in args.pipeline_batch_sizes:
                    n_batches = max(1, len(names) // bs)
                    feat_t, inf_t, tot_t = [], [], []
                    wall0 = time.perf_counter()
                    k = 0
                    while True:
                        start = (k % n_batches) * bs
                        batch = names[start:start + bs]
                        t0 = time.perf_counter()
                        Xb = featurizer.transform(batch)
                        t1 = time.perf_counter()
                        backend.predict(Xb)
                        t2 = time.perf_counter()
                        if k >= 3:                    # the first calls are warm-up
                            feat_t.append(t1 - t0); inf_t.append(t2 - t1); tot_t.append(t2 - t0)
                        k += 1
                        elapsed = time.perf_counter() - wall0
                        if len(tot_t) >= 3 and (elapsed >= args.max_seconds
                                                or (len(tot_t) >= 30 and elapsed >= args.min_seconds)):
                            break
                    tot = np.asarray(tot_t)
                    rows["pipeline"].append({
                        **ident, "batch_size": bs, "calls": len(tot_t),
                        **latency_summary(tot),
                        "featurize_ms_median": float(np.median(feat_t)) * 1e3,
                        "inference_ms_median": float(np.median(inf_t)) * 1e3,
                        "featurize_share": float(np.sum(feat_t) / np.sum(tot_t)),
                        "throughput_per_s": bs * len(tot_t) / float(np.sum(tot_t)),
                    })
                    print(f"[{args.tag}] pipeline {model:12s} {backend_name:13s} bs={bs:5d} "
                          f"p50 {rows['pipeline'][-1]['latency_ms_p50']:9.3f} ms "
                          f"(features {rows['pipeline'][-1]['featurize_share']:.0%})", flush=True)
            del backend
            # Written after every backend: an out-of-memory kill cannot be
            # caught, and must not take the finished measurements with it.
            for suite, data in rows.items():
                if data:
                    pd.DataFrame(data).to_csv(args.out_dir / f"{suite}_{args.tag}.csv", index=False)

    print(f"[{args.tag}] done", flush=True)


if __name__ == "__main__":
    main()
