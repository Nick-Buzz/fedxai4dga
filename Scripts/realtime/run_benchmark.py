"""Near real-time inference evaluation of the DGA detectors -- classification only.

No explanation (SHAP) is computed anywhere in this benchmark.  It answers,
with measurements on this machine:

  1. model      How long does one classification take, from a ready feature
                vector, per model, backend, device, batch size and CPU budget?
  2. features   How long does turning a raw query name into that vector take,
                and how does it scale with processes?
  3. pipeline   Raw name -> verdict, per stage, per batch size.
  4. stream     Under live Poisson traffic with micro-batching, what latency
                does a query see, and what query rate can be sustained on how
                many cores?
  5. checks     Load time, first-call latency, and that every serving path
                returns the probabilities (and the accuracy) already reported.

Every (device, CPU budget, model) point runs in its own process so thread
pools are sized correctly and allocator memory does not accumulate.

    python -m Scripts.realtime.run_benchmark                 # full run, ~1.5 h
    python -m Scripts.realtime.run_benchmark --quick         # ~10 min check
    python -m Scripts.realtime.report Results/realtime/<run> # figures and tables

Run on AC power with other load closed: a laptop on battery clocks down, and
the CPU numbers would say more about the power plan than about the models.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.realtime.backends import KERAS_MODELS  # noqa: E402

ALL_MODELS = list(KERAS_MODELS) + ["xgboost"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SWEEP_BATCH_SIZES = [1, 8, 64, 256, 1024, 4096]
SWEEP_THREADS = [1, 2, 4, 8, 12]
SWEEP_MODELS = ["exat_mlp", "plain_mlp", "xgboost"]
STREAM_POLICIES = ["1:0", "8:1", "32:2", "128:5", "512:20"]
STREAM_SHARDS = [1, 2, 4, 6, 8, 10]
SCALING_POLICY = "32:2"
ATTENTION_MODELS = ("exat_mlp", "single_head")
ATTENTION_MAX_BATCH = 4096


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--test-file", type=Path, default=PROJECT_ROOT / "Data/Processed/test_data.csv")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "Results/full_run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", nargs="+",
                   default=["env", "export", "features", "cpu", "gpu", "threads", "stream", "scaling"],
                   choices=["env", "export", "features", "cpu", "gpu", "threads", "stream", "scaling"])
    p.add_argument("--quick", action="store_true", help="short timings and few points, to check the setup")
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                   help="restrict the cpu, gpu and threads steps to these models")
    return p.parse_args()


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def environment() -> dict:
    import numpy
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": sh(f"git -C '{PROJECT_ROOT}' rev-parse HEAD"),
        "git_branch": sh(f"git -C '{PROJECT_ROOT}' rev-parse --abbrev-ref HEAD"),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "wsl": "microsoft" in platform.release().lower(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "cpu_model": sh("lscpu | grep 'Model name' | cut -d: -f2 | xargs"),
        "logical_cpus": os.cpu_count(),
        "cpus_available": len(os.sched_getaffinity(0)),
        "cores_per_socket": sh("lscpu | grep 'Core(s) per socket' | cut -d: -f2 | xargs"),
        "threads_per_core": sh("lscpu | grep 'Thread(s) per core' | cut -d: -f2 | xargs"),
        "memory_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    }
    for module in ("tensorflow", "onnxruntime", "xgboost", "sklearn", "pandas"):
        info[module] = sh(f"{sys.executable} -c 'import {module} as m; print(m.__version__)' 2>/dev/null")
    return info


class Runner:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.log = open(out_dir / "run.log", "a", encoding="utf-8")
        self.failures = []

    def __call__(self, label: str, args: list[str], env: dict | None = None) -> bool:
        full_env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONUNBUFFERED": "1", **(env or {})}
        t0 = time.time()
        print(f"--> {label}", flush=True)
        self.log.write(f"\n===== {datetime.now():%H:%M:%S} {label}\n{' '.join(args)}\n")
        self.log.flush()
        proc = subprocess.Popen([sys.executable, "-m", *args], cwd=PROJECT_ROOT, env=full_env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            self.log.write(line)
            if line.startswith("[") or line.startswith("  "):
                print("    " + line.rstrip(), flush=True)
        proc.wait()
        self.log.flush()
        ok = proc.returncode == 0
        print(f"    {'ok' if ok else 'FAILED (exit %d)' % proc.returncode} in {time.time() - t0:.0f}s", flush=True)
        if not ok:
            self.failures.append(label)
        return ok


def main() -> None:
    args = parse_args()
    out = args.out_dir or PROJECT_ROOT / "Results/realtime" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    run = Runner(out)
    ncpu = len(os.sched_getaffinity(0))
    cpu_env = {"CUDA_VISIBLE_DEVICES": "-1"}                 # oneDNN stays on: the CPU default
    gpu_env = {"TF_ENABLE_ONEDNN_OPTS": "0"}                 # oneDNN rewrites are CPU-only
    common = ["--seed", str(args.seed), "--results-dir", str(args.results_dir),
              "--test-file", str(args.test_file), "--out-dir", str(out)]

    batch_sizes = [1, 16, 256, 4096] if args.quick else BATCH_SIZES
    sweep_bs = [1, 256] if args.quick else SWEEP_BATCH_SIZES
    timing = ["--min-seconds", "0.3", "--max-seconds", "1.5"] if args.quick else []
    stream_timing = (["--target-queries", "600", "--min-duration", "1", "--max-duration", "2"]
                     if args.quick else [])

    if "env" in args.steps:
        env = environment()
        env.update({"quick": args.quick, "seed": args.seed, "test_file": str(args.test_file),
                    "note": "classification only; no SHAP or other explanation is computed"})
        (out / "environment.json").write_text(json.dumps(env, indent=2))
        print(json.dumps({k: env[k] for k in ("cpu_model", "cpus_available", "memory_gb", "gpu",
                                              "tensorflow", "onnxruntime", "xgboost")}, indent=2))

    if "export" in args.steps:
        run("export Keras models to ONNX",
            ["Scripts.realtime.backends", "--models", *KERAS_MODELS, "--seed", str(args.seed),
             "--results-dir", str(args.results_dir)], cpu_env)

    if "features" in args.steps:
        extra = ["--profile-domains", "3000", "--scaling-domains", "20000",
                 "--processes", "1", "4", str(ncpu)] if args.quick else []
        run("feature extraction: stages and process scaling",
            ["Scripts.realtime.featurizer_bench", "--test-file", str(args.test_file),
             "--out-dir", str(out), *extra], cpu_env)

    worker = ["Scripts.realtime.worker"]
    if "cpu" in args.steps:
        for model in args.models:
            # Attention scores are batch x heads x 49 x 49 floats: about 1 GB per
            # tensor for ExAt-MLP (12 heads) at 8192 rows, past a 7.6 GB
            # limit once ONNX Runtime holds several.  An OOM kill is uncatchable.
            sizes = [b for b in batch_sizes if b <= ATTENTION_MAX_BATCH] \
                if model in ATTENTION_MODELS else batch_sizes
            run(f"CPU, all {ncpu} logical CPUs: {model}",
                [*worker, "--device", "cpu", "--threads", str(ncpu), "--tag", f"cpu{ncpu}_{model}",
                 "--models", model, "--batch-sizes", *map(str, sizes),
                 "--suites", "latency", "coldstart", "fidelity", "fulltest", "pipeline",
                 *common, *timing], cpu_env)

    if "gpu" in args.steps:
        for model in args.models:
            # Same cap on the 8 GB GPU: the BFC allocator keeps XLA's 8192-row
            # peak and the next backend then runs out of memory.
            sizes = [b for b in batch_sizes if b <= ATTENTION_MAX_BATCH] \
                if model in ATTENTION_MODELS else batch_sizes
            run(f"GPU: {model}",
                [*worker, "--device", "gpu", "--threads", str(ncpu), "--tag", f"gpu_{model}",
                 "--models", model, "--batch-sizes", *map(str, sizes),
                 "--suites", "latency", "coldstart", "fidelity", "fulltest", "pipeline",
                 *common, *timing], gpu_env)

    if "threads" in args.steps:
        threads = [1, 4] if args.quick else [t for t in SWEEP_THREADS if t < ncpu]
        for t in threads:
            for model in [m for m in SWEEP_MODELS if m in args.models]:
                run(f"CPU budget {t} thread(s): {model}",
                    [*worker, "--device", "cpu", "--threads", str(t), "--tag", f"cpu{t}_{model}",
                     "--models", model, "--backends", "tf-function", "onnxruntime", "xgb-inplace",
                     "--batch-sizes", *map(str, sweep_bs), "--suites", "latency",
                     *common, *timing], cpu_env)

    stream_common = ["--results-dir", str(args.results_dir), "--test-file", str(args.test_file),
                     "--out-dir", str(out), "--seed", str(args.seed), *stream_timing]
    stream_models = [("exat_mlp", "onnxruntime", "cpu"), ("plain_mlp", "onnxruntime", "cpu"),
                     ("xgboost", "xgb-inplace", "cpu"), ("exat_mlp", "tf-xla", "gpu"),
                     ("xgboost", "xgb-inplace", "gpu")]
    if "stream" in args.steps:
        policies = ["1:0", "32:2"] if args.quick else STREAM_POLICIES
        for model, backend, device in stream_models:
            run(f"stream, 1 shard: {model}/{backend}/{device}",
                ["Scripts.realtime.stream", "--model", model, "--backend", backend, "--device", device,
                 "--shards", "1", "--threads-per-shard", "1" if device == "cpu" else "2",
                 "--policies", *policies, "--tag", f"policy_{model}_{device}", *stream_common],
                gpu_env if device == "gpu" else cpu_env)

    if "scaling" in args.steps:
        shards = [2, 4] if args.quick else [s for s in STREAM_SHARDS if s <= ncpu]
        for model, backend, device in stream_models:
            if device != "cpu":
                continue
            for n in shards:
                run(f"stream, {n} shards: {model}/{backend}",
                    ["Scripts.realtime.stream", "--model", model, "--backend", backend,
                     "--shards", str(n), "--threads-per-shard", "1", "--policies", SCALING_POLICY,
                     "--min-rate", str(500 * n), "--tag", f"scale_{model}_x{n}", *stream_common],
                    cpu_env)

    print(f"\nresults in {out}")
    if run.failures:
        print("FAILED steps:\n  " + "\n  ".join(run.failures))
    (out / "failures.json").write_text(json.dumps(run.failures, indent=2))


if __name__ == "__main__":
    main()
