"""Cost of turning a raw domain name into a model row.

Model-only benchmarks start from a ready feature vector; a sensor starts from
a query name.  Two measurements:

  stages   per-domain time of each step of ``DomainFeaturizer``, single thread,
           so the dominant step is identified
  scaling  domains/s with P worker processes (Python feature code holds the
           GIL, so processes rather than threads)

    python -m Scripts.realtime.featurizer_bench --out-dir Results/realtime/<run>
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.Preprocessing import feature_extractor as fx  # noqa: E402
from Scripts.realtime.featurizer import _CHARS, _KEEP, DomainFeaturizer  # noqa: E402


def stage_profile(featurizer: DomainFeaturizer, domains: list[str]) -> pd.DataFrame:
    """Time each step for every domain; one row per domain, times in microseconds."""
    ns = time.perf_counter_ns
    records = []
    wl = featurizer.whitelist
    for domain in domains:
        t0 = ns()
        prefix = featurizer.prefix(domain) or domain.lower()
        t1 = ns()
        special = fx.find_special_char_frequency(prefix)
        integers = fx.find_integer_frequency(prefix)
        vowels = fx.find_vowel_frequency(prefix)
        lexical = [fx.find_length(prefix), fx.find_max_digit_sequence(prefix),
                   fx.find_max_string_sequence(prefix), *[prefix.count(c) for c in _CHARS],
                   special, fx.find_ratio_special_char(prefix, special), integers,
                   fx.find_integer_ratio(prefix, integers), vowels,
                   fx.find_vowels_ratio(prefix, vowels), fx.find_maximum_gap_between_dots(prefix)]
        t2 = ns()
        reputation = fx.find_reputation(prefix, wl)
        t3 = ns()
        words = [fx.find_words_number(prefix), fx.find_words_mean_length(prefix)]
        t4 = ns()
        entropy = fx.get_shannon_entropy(prefix)
        t5 = ns()
        raw = np.asarray(lexical + [reputation] + words + [entropy], dtype=np.float64)[_KEEP]
        _ = ((raw - featurizer.minimum) * featurizer.inv_span).astype(np.float32)
        t6 = ns()
        records.append((len(domain), (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3,
                        (t4 - t3) / 1e3, (t5 - t4) / 1e3, (t6 - t5) / 1e3, (t6 - t0) / 1e3))
    return pd.DataFrame(records, columns=["domain_length", "suffix_strip", "lexical_counts",
                                          "ngram_reputation", "word_segmentation",
                                          "entropy", "scaling", "total"])


_WORKER_FEATURIZER = None


def _init_worker(fast=False):
    global _WORKER_FEATURIZER
    warnings.simplefilter("ignore", RuntimeWarning)
    _WORKER_FEATURIZER = DomainFeaturizer(fast=fast)


def _transform_chunk(chunk):
    return _WORKER_FEATURIZER.transform(chunk).shape[0]


def parallel_scaling(domains: list[str], processes: list[int], fast: bool = False,
                     chunk: int = 1000) -> list[dict]:
    ctx = mp.get_context("fork")
    rows = []
    chunks = [domains[i:i + chunk] for i in range(0, len(domains), chunk)]
    for p in processes:
        with ctx.Pool(p, initializer=_init_worker, initargs=(fast,)) as pool:
            pool.map(_transform_chunk, chunks[:p], chunksize=1)          # warm every worker
            t0 = time.perf_counter()
            done = sum(pool.imap_unordered(_transform_chunk, chunks, chunksize=1))
            wall = time.perf_counter() - t0
        variant = "single_segmentation" if fast else "reference"
        rows.append({"variant": variant, "processes": p, "domains": done, "seconds": wall,
                     "throughput_per_s": done / wall})
        print(f"  featurizer {variant:19s} x{p:2d} processes: {done / wall:>10,.0f} domains/s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-file", type=Path, default=PROJECT_ROOT / "Data/Processed/test_data.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--profile-domains", type=int, default=20_000)
    ap.add_argument("--scaling-domains", type=int, default=120_000)
    ap.add_argument("--processes", nargs="+", type=int, default=[1, 2, 4, 6, 8, 10, 12, 14, 16, 20])
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    names = pd.read_csv(args.test_file, usecols=["Name", "Label"])
    featurizer = DomainFeaturizer()
    featurizer.transform(names["Name"].head(500).tolist())       # warm caches

    sample = names.sample(n=args.profile_domains, random_state=0)
    profile = stage_profile(featurizer, sample["Name"].tolist())
    profile["label"] = sample["Label"].to_numpy()
    profile.to_csv(args.out_dir / "featurizer_stages_raw.csv.gz", index=False)
    stages = profile.drop(columns=["domain_length", "label"])
    summary = pd.DataFrame({
        "mean_us": stages.mean(), "median_us": stages.median(),
        "p99_us": stages.quantile(0.99), "share_of_total": stages.mean() / stages["total"].mean(),
    })
    summary.to_csv(args.out_dir / "featurizer_stages.csv", index_label="stage")
    print(summary.round(2).to_string(), flush=True)

    # Reference (the code that built the dataset) against the single-segmentation
    # variant: time on the same domains, and require identical output.
    variants = []
    outputs = {}
    for fast in (False, True):
        f = DomainFeaturizer(fast=fast)
        f.transform(sample["Name"].head(300).tolist())
        t0 = time.perf_counter()
        outputs[fast] = f.transform(sample["Name"].tolist())
        wall = time.perf_counter() - t0
        variants.append({"variant": "single_segmentation" if fast else "reference",
                         "domains": len(sample), "us_per_domain": wall / len(sample) * 1e6,
                         "throughput_per_s": len(sample) / wall})
    identical = bool(np.array_equal(outputs[False], outputs[True]))
    for v in variants:
        v["identical_to_reference"] = identical
        print(f"  featurizer {v['variant']:19s} one thread: {v['us_per_domain']:7.1f} us/domain "
              f"(identical output: {identical})", flush=True)
    pd.DataFrame(variants).to_csv(args.out_dir / "featurizer_variants.csv", index=False)

    domains = names["Name"].head(args.scaling_domains).tolist()
    cpu_count = len(__import__("os").sched_getaffinity(0))
    processes = [p for p in args.processes if p <= cpu_count]
    rows = parallel_scaling(domains, processes) + parallel_scaling(domains, processes, fast=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "featurizer_scaling.csv", index=False)


if __name__ == "__main__":
    main()
