"""Latency measurement and summary statistics shared by every suite."""

from __future__ import annotations

import gc
import time

import numpy as np
import psutil

PERCENTILES = (50, 90, 95, 99, 99.9)


def latency_summary(seconds: np.ndarray, prefix: str = "latency_ms") -> dict:
    """Percentiles, mean, std and max of a latency sample, in milliseconds."""
    ms = np.asarray(seconds, dtype=np.float64) * 1e3
    out = {f"{prefix}_p{str(p).replace('.', '_')}": float(np.percentile(ms, p)) for p in PERCENTILES}
    out.update({f"{prefix}_mean": float(ms.mean()), f"{prefix}_std": float(ms.std()),
                f"{prefix}_min": float(ms.min()), f"{prefix}_max": float(ms.max())})
    return out


def measure(fn, pool: np.ndarray, batch_size: int, *, warmup: int = 5,
            min_calls: int = 30, max_calls: int = 3000, min_seconds: float = 1.0,
            max_seconds: float = 6.0) -> dict:
    """Time repeated ``fn(batch)`` calls over consecutive, distinct batches.

    Stops once both ``min_calls`` and ``min_seconds`` are reached, or at
    ``max_seconds`` provided at least 3 calls were made, so that very large
    batches on one core still finish.  Throughput is total rows over total wall
    time (sustained), not batch size over median latency.  CPU seconds cover
    every thread of this process, so ``cpu_cores_busy`` shows how many cores a
    configuration actually kept busy.
    """
    n_batches = max(1, len(pool) // batch_size)

    def batch(i):
        start = (i % n_batches) * batch_size
        return pool[start:start + batch_size]

    for i in range(warmup):
        fn(batch(i))
    gc.collect()

    proc = psutil.Process()
    cpu0 = proc.cpu_times()
    wall0 = time.perf_counter()
    latencies = []
    i = 0
    while True:
        b = batch(warmup + i)
        t0 = time.perf_counter()
        fn(b)
        latencies.append(time.perf_counter() - t0)
        i += 1
        elapsed = time.perf_counter() - wall0
        if i >= max_calls or (i >= min_calls and elapsed >= min_seconds) \
                or (i >= 3 and elapsed >= max_seconds):
            break
    wall = time.perf_counter() - wall0
    cpu1 = proc.cpu_times()
    cpu = (cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)
    rows = i * batch_size
    lat = np.asarray(latencies)
    return {
        "calls": i,
        **latency_summary(lat),
        "throughput_per_s": rows / wall,
        "us_per_domain": float(np.median(lat)) * 1e6 / batch_size,
        "cpu_cores_busy": cpu / wall,
        "cpu_us_per_domain": cpu * 1e6 / rows,
        "rss_mb": proc.memory_info().rss / 2**20,
    }
