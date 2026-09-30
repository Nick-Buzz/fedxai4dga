"""Random hyperparameter search, selected on validation ROC-AUC.

  * Selection is on VALIDATION ROC-AUC. The test split is never touched here.
  * The proposed model is tuned. Its ABLATIONS inherit its configuration
    unchanged -- an ablation asks "what happens when I remove this component",
    which is only answerable if nothing else moved.
  * Independent BASELINES (plain_mlp) are tuned separately with the same trial
    budget; XGBoost is selected on validation inside run_experiments.py.
  * Search runs on a subsample with a short epoch budget. Relative ranking of
    configurations is stable well before absolute performance converges, and
    this keeps the search to ~30 minutes rather than a day.

    python -m Scripts.tune --model exat_mlp  --trials 12
    python -m Scripts.tune --model plain_mlp --trials 12

Writes Results/tuning/<model>/best_hyperparameters.json, which
run_experiments.py loads with --hyperparameters.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.run_experiments import load_data, make_splits, set_seed  # noqa: E402

# Search space for the attention module and the dense head.
SPACE = {
    "d_model":        [32, 64, 128],
    "num_heads":      [4, 8, 16],
    "hidden_layers":  [(400, 200, 100), (500, 250, 100), (600, 300, 150)],
    "dropout_rate":   [0.1, 0.2, 0.3, 0.4],
    "learning_rate":  [1e-4, 3e-4, 1e-3, 3e-3],
}
# Parameters that do not apply once the token branch is removed.
TOKEN_ONLY = {"d_model", "num_heads", "key_dim"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="exat_mlp",
                   choices=["exat_mlp", "no_attention", "single_head", "plain_mlp"])
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--validation-size", type=float, default=0.2)
    p.add_argument("--train-file", type=Path, default=Path("Data/Processed/train_data.csv"))
    p.add_argument("--test-file", type=Path, default=Path("Data/Processed/test_data.csv"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def sample_config(rng, model: str) -> dict:
    config = {k: rng.choice(v) for k, v in SPACE.items()}
    config["hidden_layers"] = tuple(config["hidden_layers"])
    if model == "plain_mlp":
        for key in TOKEN_ONLY:
            config.pop(key, None)
    if model == "single_head":
        config["num_heads"] = 1
    return config


def main():
    args = parse_args()
    out = args.output_dir or Path("Results/tuning") / args.model
    out.mkdir(parents=True, exist_ok=True)

    from Models.exat_mlp import VARIANTS, ExAtMLP

    feature_names, X, y, X_test, y_test, _ = load_data(args)
    set_seed(args.seed)
    X_tr, y_tr, X_val, y_val, _ = make_splits(X, y, args.seed, args)
    print(f"tuning {args.model}: {len(y_tr):,} train / {len(y_val):,} validation rows "
          f"({args.subsample:.0%} subsample)\n", flush=True)

    rng = random.Random(args.seed)
    seen, trials, best, best_auc = set(), [], None, -np.inf

    for i in range(args.trials):
        for _ in range(50):                      # avoid re-running an identical config
            config = sample_config(rng, args.model)
            key = json.dumps(config, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                break

        set_seed(args.seed)
        kwargs = dict(VARIANTS[args.model]); kwargs.update(config)
        wrapper = ExAtMLP(**kwargs)
        wrapper.build(features_number=len(feature_names))
        params = int(sum(np.prod(v.shape) for v in wrapper.model.trainable_variables))

        started = datetime.now()
        wrapper.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs,
                    batch_size=args.batch_size, patience=args.patience, verbose=0)
        auc = float(roc_auc_score(y_val, wrapper.model.predict(
            X_val, batch_size=4096, verbose=0).reshape(-1)))
        elapsed = (datetime.now() - started).total_seconds()

        record = {**{k: str(v) for k, v in config.items()},
                  "val_roc_auc": auc, "trainable_parameters": params,
                  "seconds": round(elapsed, 1)}
        trials.append(record)
        marker = ""
        if auc > best_auc:
            best, best_auc, marker = config, auc, "  <- best so far"
        print(f"  trial {i+1:>2}/{args.trials}  val AUC {auc:.5f}  "
              f"{params:>7,} params  {elapsed/60:4.1f} min{marker}", flush=True)
        pd.DataFrame(trials).to_csv(out / "all_trials.csv", index=False)

        import tensorflow as tf
        tf.keras.backend.clear_session()

    payload = {
        "model": args.model,
        "selected": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
        "val_roc_auc": best_auc,
        "selected_on": "validation ROC-AUC; the test split was not used",
        "trials": args.trials,
        "search_subsample": args.subsample,
        "search_epochs": args.epochs,
        "search_seed": args.seed,
        "search_space": {k: [list(x) if isinstance(x, tuple) else x for x in v]
                         for k, v in SPACE.items()},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Ablations of the proposed model inherit this configuration "
                "unchanged. Independent baselines are tuned separately with the "
                "same trial budget.",
    }
    (out / "best_hyperparameters.json").write_text(json.dumps(payload, indent=2),
                                                   encoding="utf-8")
    print(f"\nbest val AUC {best_auc:.5f}")
    for k, v in payload["selected"].items():
        print(f"  {k:<16} {v}")
    print(f"\nwrote {out/'best_hyperparameters.json'}")
    print("Pass it to the full run with:")
    print(f"  --hyperparameters {out/'best_hyperparameters.json'}")


if __name__ == "__main__":
    main()
