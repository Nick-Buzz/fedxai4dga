"""Recover the training min-max scaler and verify the online featurizer.

The processed CSVs hold only scaled features, and the scaler was never saved.
It is recovered here from ``Data/Raw/labeled_dataset_features.csv`` (unscaled)
by replaying ``Preprocessing.split_dataset`` -- ``train_test_split(test_size=0.2,
random_state=2345, shuffle=True)`` -- and taking the per-column minimum and
maximum of the training split, as ``Preprocessing.scale_dataset`` did.

Three checks guard against a silently different pipeline:

  1. the replayed test split lists the same domains, in the same order, as
     ``Data/Processed/test_data.csv``;
  2. scaling the raw test rows with the recovered scaler reproduces the stored
     test features;
  3. ``DomainFeaturizer`` run on raw domain names (suffix stripping, feature
     extraction and scaling, end to end) reproduces the stored test features.

    python -m Scripts.realtime.build_preprocessor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.realtime.featurizer import (  # noqa: E402
    MODEL_FEATURES, PREPROCESSOR_FILE, DomainFeaturizer,
)

TOLERANCE = 1e-5          # float32 storage of the processed CSV


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=PROJECT_ROOT / "Data/Raw/labeled_dataset_features.csv")
    ap.add_argument("--test-file", type=Path, default=PROJECT_ROOT / "Data/Processed/test_data.csv")
    ap.add_argument("--featurizer-sample", type=int, default=20_000,
                    help="test domains to push through the online featurizer")
    args = ap.parse_args()

    print(f"reading {args.raw} ...", flush=True)
    df = pd.read_csv(args.raw)
    train, test = train_test_split(df, test_size=0.2, random_state=2345, shuffle=True)
    minimum = train[MODEL_FEATURES].min()
    maximum = train[MODEL_FEATURES].max()

    stored = pd.read_csv(args.test_file)
    if not (len(stored) == len(test) and (stored["Name"].to_numpy() == test["Name"].to_numpy()).all()):
        raise SystemExit("replayed split does not match test_data.csv row for row")
    print(f"  check 1 passed: {len(test):,} test rows, same domains in the same order")

    span = (maximum - minimum).replace(0, np.nan)
    rescaled = ((test[MODEL_FEATURES] - minimum) / span).fillna(0.0).to_numpy()
    diff = np.abs(rescaled - stored[MODEL_FEATURES].to_numpy())
    if diff.max() > TOLERANCE:
        worst = MODEL_FEATURES[int(diff.max(axis=0).argmax())]
        raise SystemExit(f"recovered scaler differs from stored features (max {diff.max():.3g} on {worst})")
    print(f"  check 2 passed: recovered scaler reproduces stored features (max |diff| {diff.max():.2e})")

    PREPROCESSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_FILE.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(args.raw.relative_to(PROJECT_ROOT)),
        "split": "train_test_split(test_size=0.2, random_state=2345, shuffle=True); train part",
        "features": MODEL_FEATURES,
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
    }, indent=2), encoding="utf-8")

    featurizer = DomainFeaturizer()
    sample = stored.sample(n=min(args.featurizer_sample, len(stored)), random_state=0)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # np.mean([]) in Words_Mean
        online = featurizer.transform(sample["Name"].tolist())
    elapsed = time.perf_counter() - t0
    diff = np.abs(online - sample[MODEL_FEATURES].to_numpy(np.float32))
    row_ok = (diff <= TOLERANCE).all(axis=1)
    per_feature = pd.Series((diff > TOLERANCE).sum(axis=0), index=MODEL_FEATURES)
    report = {
        "sample": len(sample),
        "rows_exact": int(row_ok.sum()),
        "rows_exact_fraction": float(row_ok.mean()),
        "max_abs_diff": float(diff.max()),
        "mismatching_rows_by_feature": {k: int(v) for k, v in per_feature[per_feature > 0].items()},
        "mismatch_examples": sample.loc[~row_ok, "Name"].head(15).tolist(),
        "featurizer_us_per_domain_single_thread": round(elapsed / len(sample) * 1e6, 1),
    }
    (PREPROCESSOR_FILE.parent / "featurizer_fidelity.json").write_text(json.dumps(report, indent=2))
    print(f"  check 3: {report['rows_exact']:,}/{len(sample):,} domains "
          f"({report['rows_exact_fraction']:.4%}) reproduce the stored features exactly; "
          f"{report['featurizer_us_per_domain_single_thread']} us/domain")
    if report["mismatching_rows_by_feature"]:
        print(f"    mismatches by feature: {report['mismatching_rows_by_feature']}")
        print(f"    examples: {report['mismatch_examples'][:8]}")
    print(f"written {PREPROCESSOR_FILE}")


if __name__ == "__main__":
    main()
