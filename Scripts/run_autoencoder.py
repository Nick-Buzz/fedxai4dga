"""Train and evaluate the autoencoder baseline.

    python -m Scripts.run_autoencoder --output-dir Results/full_run --seeds 42

Procedure:

  * training data: the training split oversampled with SMOTE (random_state=42,
    ``Preprocessing.oversample_data``), both classes;
  * model: ``Models.AutoEncoder.AutoencoderModel`` with its default
    configuration, trained to reconstruct its input (its own ``fit``);
  * score: per-domain reconstruction mean squared error on the test set;
  * decision: DGA when the reconstruction error exceeds ``--threshold``
    (default 0.002).

Metrics come from ``Scripts/metrics.py`` with the reconstruction error as the
continuous score, so ROC-AUC is threshold-free.  Results are written in the
same layout as run_experiments.py (``<output-dir>/autoencoder/seed_<n>/``), so
make_results_table.py picks them up.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.metrics import binary_metrics, per_family_metrics  # noqa: E402
from Scripts.run_experiments import load_data, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--threshold", type=float, default=0.002,
                   help="reconstruction MSE above which a domain is flagged as DGA")
    p.add_argument("--train-file", type=Path, default=Path("Data/Processed/train_data.csv"))
    p.add_argument("--test-file", type=Path, default=Path("Data/Processed/test_data.csv"))
    p.add_argument("--train-rows", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("Results/full_run"))
    p.add_argument("--smoke-test", action="store_true",
                   help="20,000 training rows, 5,000 test rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from Models.AutoEncoder import AutoencoderModel
    from Scripts.Preprocessing.Preprocessing import oversample_data

    feature_names, X, y, X_test, y_test, meta = load_data(args)
    X_res, _ = oversample_data(X, y)
    X_res = np.asarray(X_res, dtype=np.float32)

    for seed in args.seeds:
        run_dir = args.output_dir / "autoencoder" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== autoencoder | seed {seed} =====", flush=True)
        set_seed(seed)

        started = datetime.now()
        wrapper = AutoencoderModel()
        wrapper.build(features_number=len(feature_names))
        wrapper.fit(X_res, verbose=0 if args.smoke_test else 2)
        reconstruction = wrapper.model.predict(X_test, batch_size=4096, verbose=0)
        score = wrapper.loss(X_test, reconstruction).numpy().reshape(-1).astype(np.float64)
        elapsed = (datetime.now() - started).total_seconds()

        result = binary_metrics(y_test, score, args.threshold)
        result.pop("log_loss", None)          # the score is an error, not a probability
        result.update({"model": "autoencoder", "seed": seed, "score": "reconstruction MSE",
                       "seconds": round(elapsed, 1)})
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        per_family_metrics(meta["Family"], y_test, score, args.threshold) \
            .to_csv(run_dir / "family_metrics.csv", index=False)
        pd.DataFrame({"Name": meta["Name"], "Family": meta["Family"], "y_true": y_test,
                      "reconstruction_mse": score}) \
            .to_csv(run_dir / "test_predictions.csv.gz", index=False, compression="gzip")
        print(f"  acc {result['accuracy']:.4f} | recall {result['recall']:.4f} | "
              f"FPR {result['false_positive_rate']:.4f} | AUC {result['roc_auc']:.4f} | "
              f"{elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
