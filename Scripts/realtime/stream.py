"""Open-loop streaming simulation: can the detector keep up with live DNS traffic?

Queries arrive as a Poisson process at a fixed rate, so arrival times come
from a schedule fixed in advance and never wait for the detector.  A
closed-loop benchmark (send, wait, send) hides queueing delay: a slow
response simply delays the next request (coordinated omission).  Here a slow
batch leaves later queries waiting, and that wait is counted in their latency.

Each shard is one process holding one featurizer and one model, pinned to its
own CPUs.  It serves its queue with dynamic micro-batching:

    dispatch when  pending >= max_batch  or  oldest has waited max_wait_ms

(max_batch=1 means no batching, every query scored as it arrives).  Latency
is completion minus arrival, per query, so it includes queueing, feature
extraction from the raw name and inference.  N shards each receive rate/N,
which is exact for Poisson traffic split across instances (e.g. by hashing
the query name at a load balancer).

For each configuration the offered rate is increased until the detector stops
keeping up at two consecutive rates: its queue is not empty within 50 ms
(+ max_wait) of the last arrival, or p99 exceeds one second.  Below capacity the queue empties almost
at once; above it the backlog grows for the whole run.  The last stable rate
is the sustainable capacity.

    python -m Scripts.realtime.stream --model exat_mlp --backend onnxruntime \\
        --shards 1 --policies 1:0 32:2 256:10 --out-dir Results/realtime/<run> --tag s1
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RATES = [100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000,
                 12000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 80000, 100000]
HIST_EDGES_LOG10_MS = (-2.0, 4.0, 241)          # 0.01 ms .. 10 s, 40 bins per decade
DRAIN_SECONDS = 1.0          # serve the backlog at most this long after the last arrival
DRAIN_TOLERANCE_S = 0.05     # a stable shard empties its queue this soon (+ max_wait)
UNSTABLE_P99_MS = 1000.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--threads-per-shard", type=int, default=1)
    p.add_argument("--policies", nargs="+", default=["1:0", "32:2", "256:10"],
                   help="max_batch:max_wait_ms pairs")
    p.add_argument("--rates", nargs="+", type=float, default=DEFAULT_RATES,
                   help="total offered queries/s, swept in increasing order")
    p.add_argument("--min-rate", type=float, default=0.0, help="skip rates below this")
    p.add_argument("--target-queries", type=int, default=3000,
                   help="queries per point; duration = clip(target/rate, min, max)")
    p.add_argument("--min-duration", type=float, default=3.0)
    p.add_argument("--max-duration", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "Results/full_run")
    p.add_argument("--test-file", type=Path, default=PROJECT_ROOT / "Data/Processed/test_data.csv")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Results/realtime")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------- shard

def shard_main(shard: int, cfg: dict, tasks, results, barrier) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    import os
    from Scripts.realtime.backends import configure_threads

    if cfg["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    cpus = configure_threads(cfg["threads"], slot=shard)

    import warnings
    warnings.simplefilter("ignore")
    import numpy as np
    import pandas as pd
    from Scripts.realtime.backends import make_backend
    from Scripts.realtime.featurizer import DomainFeaturizer

    if cfg["backend"].startswith("tf") or cfg["backend"] == "keras-predict":
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.threading.set_intra_op_parallelism_threads(cfg["threads"])
        tf.config.threading.set_inter_op_parallelism_threads(1)

    names = pd.read_csv(cfg["test_file"], usecols=["Name"])["Name"].tolist()
    featurizer = DomainFeaturizer()
    backend = make_backend(cfg["model"], cfg["backend"], cfg["device"], cfg["threads"],
                           Path(cfg["results_dir"]), cfg["seed"], Path(cfg["output_root"]))
    predict = backend.predict

    def process(batch_names):
        return predict(featurizer.transform(batch_names))

    # Warm every batch shape the policies can produce, so tracing and first-use
    # allocation are not charged to the first queries of a run.
    max_batch = max(cfg["max_batches"])
    size = 1
    while True:
        for _ in range(3):
            process(names[:size])
        if size >= max_batch:
            break
        size = min(size * 2, max_batch)
    for s in range(1, min(max_batch, 64) + 1):
        process(names[:s])
    results.put(("ready", shard, cpus))

    clock = time.perf_counter
    while True:
        task = tasks.get()
        if task is None:
            break
        rate, max_batch, max_wait, duration = (task["rate"], task["max_batch"],
                                               task["max_wait_ms"] / 1e3, task["duration"])
        rng = np.random.default_rng([cfg["seed"], shard, int(task["point"])])
        gaps = rng.exponential(1.0 / rate, size=int(rate * duration * 1.3) + 32)
        arrivals = np.cumsum(gaps)
        arrivals = arrivals[arrivals < duration]
        n = len(arrivals)
        which = rng.integers(0, len(names), size=n)
        start = np.full(n, np.nan)
        done = np.full(n, np.nan)
        batch_sizes = []
        busy = 0.0

        barrier.wait()
        t0 = clock()

        def wait_until(t):
            # Sleep overshoots by up to ~0.3 ms (occasionally >1 ms) under WSL2;
            # sleeping only to 1.5 ms before the target and spinning the rest
            # keeps timer error out of the measured latency.
            remaining = t - (clock() - t0)
            if remaining > 2e-3:
                time.sleep(remaining - 1.5e-3)
            while clock() - t0 < t:
                pass

        i = 0
        stop_at = duration + DRAIN_SECONDS
        while i < n:
            now = clock() - t0
            if now > stop_at:
                break
            arrived = int(np.searchsorted(arrivals, now, side="right"))
            pending = arrived - i
            if pending <= 0:
                wait_until(min(arrivals[i], stop_at))
                continue
            if pending >= max_batch or now - arrivals[i] >= max_wait:
                k = min(pending, max_batch)
                s = clock() - t0
                process([names[j] for j in which[i:i + k]])
                e = clock() - t0
                start[i:i + k] = s
                done[i:i + k] = e
                busy += e - s
                batch_sizes.append(k)
                i += k
            else:
                fill = arrivals[i + max_batch - 1] if i + max_batch - 1 < n else np.inf
                wait_until(min(arrivals[i] + max_wait, fill, stop_at))

        served = i
        results.put(("result", shard, {
            "offered": n,
            "served": served,
            "latency": (done[:served] - arrivals[:served]).astype(np.float32),
            "queue": (start[:served] - arrivals[:served]).astype(np.float32),
            "batch_sizes": np.asarray(batch_sizes, dtype=np.int32),
            "busy": busy,
            "duration": duration,
            # Time from the last arrival until the backlog was empty.  Near zero
            # when the shard keeps up; grows with the run length when it does not.
            "drain": (float(done[served - 1] - arrivals[-1]) if served == n and n
                      else float("inf")),
            "window": max(duration, float(done[served - 1])) if served else duration,
        }))


# --------------------------------------------------------------------- driver

def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_ROOT))
    import numpy as np
    import pandas as pd
    from Scripts.realtime.timing import latency_summary

    policies = []
    for spec in args.policies:
        b, w = spec.split(":")
        policies.append((int(b), float(w)))

    cfg = {"model": args.model, "backend": args.backend, "device": args.device,
           "threads": args.threads_per_shard, "seed": args.seed,
           "results_dir": str(args.results_dir), "test_file": str(args.test_file),
           "output_root": str(args.output_root), "max_batches": [b for b, _ in policies]}
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.shards + 1)
    results = ctx.Queue()
    queues = [ctx.Queue() for _ in range(args.shards)]
    procs = [ctx.Process(target=shard_main, args=(s, cfg, queues[s], results, barrier), daemon=True)
             for s in range(args.shards)]
    for proc in procs:
        proc.start()
    cpus = {}
    for _ in procs:
        kind, shard, payload = results.get(timeout=600)
        cpus[shard] = payload
    print(f"[{args.tag}] {args.shards} shard(s) ready: {args.model}/{args.backend}/{args.device}, "
          f"{args.threads_per_shard} thread(s) each, CPUs {list(cpus.values())}", flush=True)

    edges = np.logspace(*HIST_EDGES_LOG10_MS)
    rows, hists = [], []
    point = 0
    ident = {"model": args.model, "backend": args.backend, "device": args.device,
             "shards": args.shards, "threads_per_shard": args.threads_per_shard,
             "cpus_total": len({c for v in cpus.values() for c in v})}
    for max_batch, max_wait in policies:
        misses = 0
        for rate in args.rates:
            if rate < args.min_rate:
                continue
            duration = float(np.clip(args.target_queries / rate, args.min_duration, args.max_duration))
            point += 1
            for q in queues:
                q.put({"rate": rate / args.shards, "max_batch": max_batch, "max_wait_ms": max_wait,
                       "duration": duration, "point": point})
            barrier.wait()
            parts = [results.get(timeout=duration + 600)[2] for _ in procs]

            offered = sum(p["offered"] for p in parts)
            served = sum(p["served"] for p in parts)
            lat = np.concatenate([p["latency"] for p in parts])
            queue = np.concatenate([p["queue"] for p in parts])
            batches = np.concatenate([p["batch_sizes"] for p in parts])
            stats = latency_summary(lat) if len(lat) else {}
            drain = max(p["drain"] for p in parts)
            stable = (served == offered and drain <= DRAIN_TOLERANCE_S + max_wait / 1e3
                      and stats.get("latency_ms_p99", np.inf) <= UNSTABLE_P99_MS)
            ms = lat * 1e3
            row = {**ident, "max_batch": max_batch, "max_wait_ms": max_wait,
                   "offered_rate": rate, "duration_s": duration, "offered": offered, "served": served,
                   "achieved_rate": served / duration, "stable": stable, **stats,
                   "queue_ms_p50": float(np.median(queue)) * 1e3 if len(queue) else np.nan,
                   "queue_ms_p99": float(np.percentile(queue, 99)) * 1e3 if len(queue) else np.nan,
                   "mean_batch": float(batches.mean()) if len(batches) else np.nan,
                   "drain_ms": drain * 1e3,
                   "utilization": float(np.mean([p["busy"] / p["window"] for p in parts])),
                   "within_1ms": float((ms <= 1).sum() / offered),
                   "within_10ms": float((ms <= 10).sum() / offered),
                   "within_100ms": float((ms <= 100).sum() / offered)}
            rows.append(row)
            counts, _ = np.histogram(np.clip(ms, edges[0], edges[-1]), bins=edges)
            hists.append({**ident, "max_batch": max_batch, "max_wait_ms": max_wait,
                          "offered_rate": rate, "counts": " ".join(map(str, counts))})
            print(f"[{args.tag}] {args.model:10s} x{args.shards:<2d} batch<={max_batch:<5d} "
                  f"wait<={max_wait:>4g}ms  {rate:>8,.0f} q/s  "
                  f"p50 {stats.get('latency_ms_p50', np.nan):9.3f}  p99 {stats.get('latency_ms_p99', np.nan):9.3f} ms  "
                  f"util {row['utilization']:.2f}  batch {row['mean_batch']:.1f}  "
                  f"{'ok' if stable else 'SATURATED'}", flush=True)
            # One unstable point can be a transient (a WSL or OS hiccup); two in
            # a row mark saturation.  Capacity is the highest stable rate.
            misses = 0 if stable else misses + 1
            if misses >= 2:
                break
    for q in queues:
        q.put(None)
    for proc in procs:
        proc.join(timeout=30)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / f"stream_{args.tag}.csv", index=False)
    pd.DataFrame(hists).to_csv(args.out_dir / f"stream_hist_{args.tag}.csv.gz", index=False)
    print(f"[{args.tag}] done", flush=True)


if __name__ == "__main__":
    main()
