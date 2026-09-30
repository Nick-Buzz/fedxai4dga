"""Train and evaluate every model variant across seeds, with full provenance.

  * ExAt-MLP and its ablations, which differ in exactly one component
    (``no_attention``, ``single_head``), and the plain MLP baseline
  * XGBoost, with its configuration selected on validation ROC-AUC
  * positive-class metrics, probability-based ROC-AUC, false-positive rate and
    per-family recall, via ``Scripts/metrics.py``

Validation is held out from the ORIGINAL training rows with stratification
BEFORE SMOTE, and SMOTE is fitted only on the remaining training subset.  The
test split is never resampled.

    python -m Scripts.run_experiments --models exat_mlp no_attention plain_mlp xgboost
    python -m Scripts.run_experiments --models exat_mlp --subsample 0.25   # CPU-bound path
    python -m Scripts.run_experiments --smoke-test                         # 60 seconds
"""

from __future__ import annotations

import os

# oneDNN is a CPU optimisation library whose graph rewrites are CPU-only;
# disabling it keeps GPU runs on standard kernels. Must be set before
# TensorFlow is imported.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


import argparse
import gc
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.metrics import aggregate, binary_metrics, per_family_metrics  # noqa: E402

DEFAULT_SEEDS = (42, 2024, 2345, 7, 1337)
KERAS_VARIANTS = ("exat_mlp", "no_attention", "single_head", "plain_mlp")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["exat_mlp", "no_attention", "plain_mlp"],
                   help=f"any of {KERAS_VARIANTS} plus 'xgboost'")
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--validation-size", type=float, default=0.2)
    p.add_argument("--subsample", type=float, default=1.0,
                   help="stratified fraction of the ORIGINAL training rows to keep, "
                        "when compute-bound.")
    p.add_argument("--train-rows", type=int, default=None,
                   help="read only the first N rows of the training file. Needed only "
                        "for a file that appends SMOTE rows after the original ones; "
                        "the file written by build_dataset.py has none.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--train-file", type=Path, default=Path("Data/Processed/train_data.csv"))
    p.add_argument("--test-file", type=Path, default=Path("Data/Processed/test_data.csv"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hyperparameters", type=Path, default=None,
                   help="best_hyperparameters.json from Scripts/tune.py. Applied to "
                        "the proposed model and its ablations; independent baselines "
                        "take their own file via --baseline-hyperparameters.")
    p.add_argument("--baseline-hyperparameters", type=Path, default=None,
                   help="best_hyperparameters.json for plain_mlp.")
    p.add_argument("--smoke-test", action="store_true",
                   help="20,000 training rows, 5,000 test rows, one epoch.")
    return p.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # enable_op_determinism() disables most fast kernels and slows training
    # several-fold. Seeding alone is enough for reproducible reporting; turn
    # determinism on only when bit-exactness is required.
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass


def distribution(y) -> dict:
    values, counts = np.unique(np.asarray(y).reshape(-1), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=PROJECT_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_data(args):
    train_header = pd.read_csv(args.train_file, nrows=0).columns.tolist()
    feature_names = train_header[:-1]

    # The training file must hold only ORIGINAL rows: SMOTE is applied later, to
    # the training subset only, after the validation holdout (make_splits).
    train_rows = 20_000 if args.smoke_test else getattr(args, "train_rows", None)
    train_df = pd.read_csv(
        args.train_file, usecols=feature_names + ["Label"], nrows=train_rows,
        dtype={c: np.float32 for c in feature_names} | {"Label": np.int8},
    )
    print(f"training rows: {len(train_df):,} {distribution(train_df['Label'])}", flush=True)

    test_rows = 5_000 if args.smoke_test else None
    test_df = pd.read_csv(
        args.test_file, usecols=feature_names + ["Name", "Label", "Family"],
        nrows=test_rows, dtype={c: np.float32 for c in feature_names} | {"Label": np.int8},
    )

    X = train_df[feature_names].to_numpy(dtype=np.float32, copy=False)
    y = train_df["Label"].to_numpy(dtype=np.int8, copy=False)
    X_test = test_df[feature_names].to_numpy(dtype=np.float32, copy=False)
    y_test = test_df["Label"].to_numpy(dtype=np.int8, copy=False)
    return feature_names, X, y, X_test, y_test, test_df[["Name", "Family"]].copy()


def make_splits(X, y, seed, args):
    """Stratified validation holdout BEFORE SMOTE, then SMOTE on the rest."""
    from imblearn.over_sampling import SMOTE

    if args.subsample < 1.0:
        X, _, y, _ = train_test_split(X, y, train_size=args.subsample,
                                      random_state=seed, stratify=y)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.validation_size, random_state=seed,
        shuffle=True, stratify=y,
    )
    pre = distribution(y_tr)
    X_res, y_res = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
    return (np.asarray(X_res, np.float32), np.asarray(y_res, np.int8),
            X_val, y_val, pre)


def load_hyperparameters(path):
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = payload.get("selected", payload)
    if "hidden_layers" in selected:
        selected["hidden_layers"] = tuple(selected["hidden_layers"])
    return selected


def run_keras(variant, X_tr, y_tr, X_val, y_val, X_test, n_features, seed, run_dir, args):
    import tensorflow as tf
    from Models.exat_mlp import build_variant

    # Ablations inherit the proposed model's configuration, so that removing a
    # component is the only difference. Independent baselines get their own.
    source = args.baseline_hyperparameters if variant == "plain_mlp" else args.hyperparameters
    overrides = load_hyperparameters(source)
    if variant == "single_head":
        overrides.pop("num_heads", None)
    if overrides:
        (run_dir / "hyperparameters.json").write_text(
            json.dumps({"source": str(source),
                        "applied": {k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in overrides.items()}}, indent=2),
            encoding="utf-8")

    wrapper = build_variant(variant, features_number=n_features, **overrides)
    wrapper.model.summary(print_fn=lambda s: open(run_dir / "architecture.txt", "a").write(s + "\n"))

    callbacks = [
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv"), append=args.resume),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(run_dir / "best_model.keras"),
                                           monitor="val_loss", save_best_only=True, verbose=0),
    ]
    wrapper.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs,
                batch_size=args.batch_size, patience=args.patience,
                callbacks=callbacks, verbose=2)
    wrapper.model.save(run_dir / "model.keras")

    trainable = int(sum(np.prod(v.shape) for v in wrapper.model.trainable_variables))
    probabilities = wrapper.model.predict(X_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    return probabilities, {"trainable_parameters": trainable}


def run_xgboost(X_tr, y_tr, X_val, y_val, X_test, seed, run_dir, args):
    """Fit a fixed grid and keep the configuration with the best validation ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    grid = [
        dict(n_estimators=400, max_depth=6,  learning_rate=0.10, subsample=0.9, colsample_bytree=0.9),
        dict(n_estimators=600, max_depth=8,  learning_rate=0.08, subsample=0.9, colsample_bytree=0.8),
        dict(n_estimators=800, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
        dict(n_estimators=400, max_depth=12, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7),
    ]
    if args.smoke_test:
        grid = grid[:1]
        grid[0]["n_estimators"] = 20

    trials, best, best_auc, best_cfg = [], None, -np.inf, None
    for cfg in grid:
        clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                            tree_method="hist", random_state=seed, n_jobs=-1, **cfg)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        trials.append({**cfg, "val_roc_auc": float(auc)})
        if auc > best_auc:
            best, best_auc, best_cfg = clf, auc, cfg

    pd.DataFrame(trials).to_csv(run_dir / "xgboost_trials.csv", index=False)
    best.save_model(str(run_dir / "model.json"))
    return best.predict_proba(X_test)[:, 1], {"selected_config": best_cfg,
                                              "val_roc_auc": float(best_auc),
                                              "n_trials": len(grid)}


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs, args.patience = 1, 1

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = args.output_dir or Path("Results/experiments") / stamp
    out.mkdir(parents=True, exist_ok=True)

    feature_names, X, y, X_test, y_test, meta = load_data(args)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "models": args.models,
        "seeds": args.seeds,
        "subsample": args.subsample,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "threshold": args.threshold,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "test_distribution": distribution(y_test),
        "smoke_test": args.smoke_test,
        "validation_policy": "stratified holdout before SMOTE; SMOTE on training subset only",
        "metrics_policy": "positive class = DGA; ROC-AUC from probabilities",
        "hyperparameters": str(args.hyperparameters) if args.hyperparameters else "defaults",
        "baseline_hyperparameters": (str(args.baseline_hyperparameters)
                                     if args.baseline_hyperparameters else "defaults"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            run_dir = out / model_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            if args.resume and metrics_path.exists():
                rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                print(f"[skip] {model_name} seed {seed} already complete", flush=True)
                continue

            print(f"\n===== {model_name} | seed {seed} =====", flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            set_seed(seed)

            X_tr, y_tr, X_val, y_val, pre = make_splits(X, y, seed, args)
            (run_dir / "split_distribution.json").write_text(json.dumps({
                "seed": seed,
                "train_before_smote": {"total": int(sum(pre.values())), "classes": pre},
                "train_after_smote": {"total": len(y_tr), "classes": distribution(y_tr)},
                "validation": {"total": len(y_val), "classes": distribution(y_val)},
                "test": {"total": len(y_test), "classes": distribution(y_test)},
            }, indent=2), encoding="utf-8")

            started = datetime.now()
            if model_name == "xgboost":
                probabilities, extra = run_xgboost(X_tr, y_tr, X_val, y_val, X_test,
                                                   seed, run_dir, args)
            elif model_name in KERAS_VARIANTS:
                probabilities, extra = run_keras(model_name, X_tr, y_tr, X_val, y_val,
                                                 X_test, len(feature_names), seed, run_dir, args)
            else:
                raise ValueError(f"unknown model {model_name!r}")
            elapsed = (datetime.now() - started).total_seconds()

            result = binary_metrics(y_test, probabilities, args.threshold)
            result.update({"model": model_name, "seed": seed,
                           "seconds": round(elapsed, 1), **extra})
            metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

            per_family_metrics(meta["Family"], y_test, probabilities, args.threshold) \
                .to_csv(run_dir / "family_metrics.csv", index=False)
            pd.DataFrame({"Name": meta["Name"], "Family": meta["Family"],
                          "y_true": y_test, "probability": probabilities}) \
                .to_csv(run_dir / "test_predictions.csv.gz", index=False, compression="gzip")

            rows.append(result)
            pd.DataFrame(rows).to_csv(out / "per_seed_metrics.csv", index=False)
            print(f"  acc {result['accuracy']:.4f} | recall {result['recall']:.4f} | "
                  f"FPR {result['false_positive_rate']:.4f} | AUC {result['roc_auc']:.4f} | "
                  f"{elapsed/60:.1f} min", flush=True)

            gc.collect()
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:
                pass

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_seed_metrics.csv", index=False)
    parts = []
    for model_name, group in frame.groupby("model", sort=False):
        agg = aggregate(group.to_dict("records"))
        agg.insert(0, "model", model_name)
        parts.append(agg)
    pd.concat(parts).to_csv(out / "aggregate_metrics.csv", index=False)
    (out / "completed.json").write_text(json.dumps({
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "models": args.models, "seeds": args.seeds,
    }, indent=2), encoding="utf-8")
    print(f"\nAll results in {out}", flush=True)


if __name__ == "__main__":
    main()
