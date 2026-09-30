"""Post-hoc SHAP analysis of the trained classifiers in ``Results/full_run``.

  * Explainer: the model-agnostic KernelExplainer (KernelSHAP) on each model's
    predicted probability of the DGA class.
  * Background: the training set summarised by K-means into 50 centroids, the
    eXplainable Background Instances (XBIs), weighted by cluster size.
  * Explained: 250 eXplainable Test Instances (XTIs) sub-sampled from the test set.
  * Correlated features: a pairwise Pearson correlation analysis on the training
    set groups features with |r| >= --corr-threshold, and the SHAP values within
    each group are consolidated (summed, which SHAP additivity allows).
  * Plots: summary plot of the 20 highest-ranked features (Fig. 2) and
    dependence plots coloured by the feature with the strongest interaction
    (Fig. 3, Entropy).

Additivity (base value + sum of attributions == predicted probability) is
checked for every model and seed.

    python -u -m Scripts.explain                          # MLP and ExAt-MLP
    python -u -m Scripts.explain --models exat_mlp plain_mlp xgboost
    python -u -m Scripts.explain --smoke-test             # a minute or two

Outputs go to ``Results/shap/<model>/seed_<n>/`` plus cross-model summaries in
``Results/shap/summary/``.
"""

from __future__ import annotations

import os

# See Scripts/run_experiments.py.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPENDENCE_FEATURES = ("Entropy", "Length", "Reputation", "Words_Freq", "Words_Mean")
SAMPLE_SEED = 1452
ADDITIVITY_ATOL = 1e-3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["plain_mlp", "exat_mlp"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--results-dir", type=Path, default=Path("Results/full_run"))
    p.add_argument("--output-dir", type=Path, default=Path("Results/shap"))
    p.add_argument("--train-file", type=Path, default=Path("Data/Processed/train_data.csv"))
    p.add_argument("--test-file", type=Path, default=Path("Data/Processed/test_data.csv"))
    p.add_argument("--n-background", type=int, default=50,
                   help="K-means centroids summarising the training set (XBIs)")
    p.add_argument("--n-explain", type=int, default=250,
                   help="test instances to explain (XTIs)")
    p.add_argument("--nsamples", type=int, default=2048,
                   help="KernelSHAP coalitions per explained domain")
    p.add_argument("--corr-threshold", type=float, default=0.7,
                   help="|Pearson r| at or above which features are consolidated")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="prediction batch; lower it if the GPU runs out of memory")
    p.add_argument("--train-rows", type=int, default=None,
                   help="read only the first N rows of the training file (see "
                        "run_experiments.py --train-rows)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="first 3,000 rows of each file and small SHAP settings")
    return p.parse_args()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- data

def load_sets(args):
    """XBIs (K-means centroids of the training set), XTIs and feature groups."""
    import shap
    header = pd.read_csv(args.train_file, nrows=0).columns.tolist()
    features = header[:-1]
    rows = 3000 if args.smoke_test else None
    train = pd.read_csv(args.train_file, usecols=features, nrows=rows or args.train_rows,
                        dtype={c: np.float32 for c in features})
    test = pd.read_csv(args.test_file, usecols=features + ["Name", "Label", "Family"],
                       nrows=rows, dtype={c: np.float32 for c in features})

    t0 = time.time()
    background = shap.kmeans(train[features].to_numpy(np.float64), args.n_background)
    print(f"background: {args.n_background} K-means centroids of {len(train):,} "
          f"training instances ({time.time() - t0:.0f}s)")
    explained = shap.utils.sample(test, min(args.n_explain, len(test)),
                                  random_state=SAMPLE_SEED).reset_index(drop=True)
    print(f"explained: {len(explained)} test instances sampled from {len(test):,}")
    groups = correlated_groups(train[features], args.corr_threshold)
    return features, background, explained, groups


def correlated_groups(frame: pd.DataFrame, threshold: float) -> list[list[str]]:
    """Connected components of the graph linking features with |r| >= threshold."""
    corr = frame.corr().abs().fillna(0.0).to_numpy()
    names = list(frame.columns)
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(len(names)), 2):
        if corr[i, j] >= threshold:
            parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        groups.setdefault(find(i), []).append(name)
    return list(groups.values())


# ------------------------------------------------------------------------- models

def load_keras(path):
    import tensorflow as tf
    import Models.exat_mlp  # noqa: F401  registers FeatureTokenizer for deserialisation
    return tf.keras.models.load_model(path, compile=False)


def load_predictor(model_name, run_dir, batch_size):
    """P(DGA) as a function of a feature matrix."""
    if model_name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(run_dir / "model.json")
        return lambda X: model.predict_proba(np.asarray(X, np.float32))[:, 1].astype(np.float64)
    model = load_keras(run_dir / "model.keras")
    return lambda X: model.predict(np.asarray(X, np.float32), batch_size=batch_size,
                                   verbose=0).reshape(-1).astype(np.float64)


def to_positive_class(values, n, d):
    """Normalise every SHAP output layout to (n, d) for the DGA class."""
    if isinstance(values, list):
        if len(values) == 1:
            values = values[0]
        elif len(values) == 2:
            values = values[1]
        else:
            raise ValueError(f"unexpected list of {len(values)} outputs")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        if values.shape[-1] == 1:
            values = values[..., 0]
        elif values.shape[-1] == 2:
            values = values[..., 1]
        else:
            raise ValueError(f"unexpected SHAP shape {values.shape}")
    if values.shape != (n, d):
        raise ValueError(f"SHAP values have shape {values.shape}, expected {(n, d)}")
    return values


def scalar_base(expected_value):
    ev = np.asarray(expected_value, dtype=np.float64).reshape(-1)
    return float(ev[-1])   # single output -> that value; two outputs -> DGA class


def explain_one(model_name, seed, features, background, explained, args):
    import shap
    run_dir = args.results_dir / model_name / f"seed_{seed}"
    Xe = explained[features].to_numpy(np.float64)
    n, d = Xe.shape
    t0 = time.time()

    predict = load_predictor(model_name, run_dir, args.batch_size)
    explainer = shap.KernelExplainer(predict, background)
    np.random.seed(seed)   # KernelSHAP samples coalitions with np.random
    raw = explainer.shap_values(Xe, nsamples=args.nsamples,
                                l1_reg=f"num_features({d})", silent=True)
    method = f"KernelExplainer(nsamples={args.nsamples}, l1_reg=num_features({d}))"

    values = to_positive_class(raw, n, d)
    base = scalar_base(explainer.expected_value)
    proba = predict(Xe)
    gap = np.abs(base + values.sum(axis=1) - proba)
    print(f"  {model_name} seed {seed}: {method}, {time.time() - t0:.0f}s, "
          f"base {base:.4f}, max additivity gap {gap.max():.2e}")
    if gap.max() > ADDITIVITY_ATOL:
        raise AssertionError(f"{model_name} seed {seed}: additivity broken, "
                             f"max gap {gap.max():.3e} > {ADDITIVITY_ATOL}")
    return values, base, proba, {"method": method, "seconds": round(time.time() - t0, 1),
                                 "base_value": base,
                                 "max_additivity_gap": float(gap.max())}


# -------------------------------------------------------------------------- plots

def plot_run(values, explained, features, out_dir, title):
    import matplotlib
    matplotlib.use("Agg")
    import warnings
    import matplotlib.pyplot as plt
    import shap
    warnings.filterwarnings("ignore", category=FutureWarning)   # shap's plotting RNG notice
    X = explained[features].reset_index(drop=True)

    # Fig. 2: summary plot of the 20 highest-ranked features
    shap.summary_plot(values, X, show=False, max_display=20)
    plt.title(f"{title}: SHAP on P(DGA)")
    plt.tight_layout(); plt.savefig(out_dir / "summary.png", dpi=150); plt.close("all")

    shap.summary_plot(values, X, plot_type="bar", show=False, max_display=20)
    plt.title(f"{title}: mean |SHAP|")
    plt.tight_layout(); plt.savefig(out_dir / "bar.png", dpi=150); plt.close("all")

    # Fig. 3: dependence plots, coloured by the most strongly interacting feature
    for f in DEPENDENCE_FEATURES:
        if f in features:
            shap.dependence_plot(f, values, X, interaction_index="auto", show=False)
            plt.title(f"{title}: {f}", fontsize=9)
            plt.tight_layout(); plt.savefig(out_dir / f"dependence_{f}.png", dpi=150)
            plt.close("all")


# ------------------------------------------------------------------------ summary

def importance_table(values, features):
    imp = pd.DataFrame({"feature": features,
                        "mean_abs_shap": np.abs(values).mean(axis=0),
                        "mean_shap": values.mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def consolidated_table(values, features, groups):
    """Importance after summing the SHAP values of each correlated group."""
    rows = []
    for group in groups:
        idx = [features.index(f) for f in group]
        summed = values[:, idx].sum(axis=1)
        rows.append({"group": " + ".join(group), "size": len(group),
                     "mean_abs_shap": float(np.abs(summed).mean()),
                     "mean_shap": float(summed.mean())})
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def summarise(all_imp, features, groups, out_dir):
    from scipy.stats import spearmanr
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"group": [" + ".join(g) for g in groups if len(g) > 1]}) \
        .to_csv(out_dir / "correlated_groups.csv", index=False)

    ranks = []
    for model, by_seed in all_imp.items():
        for f in features:
            r = [int(by_seed[s].set_index("feature").loc[f, "rank"]) for s in sorted(by_seed)]
            ranks.append({"model": model, "feature": f, "rank_mean": np.mean(r),
                          "ranks": " ".join(map(str, r))})
    rk = pd.DataFrame(ranks)
    rk.to_csv(out_dir / "feature_ranks.csv", index=False)

    models = list(all_imp)
    avg = {m: np.mean([all_imp[m][s].set_index("feature").loc[features, "mean_abs_shap"]
                       for s in all_imp[m]], axis=0) for m in models}
    cross = [{"model_a": a, "model_b": b, "spearman": spearmanr(avg[a], avg[b]).correlation,
              "top20_overlap": len(set(np.array(features)[np.argsort(-avg[a])[:20]]) &
                                   set(np.array(features)[np.argsort(-avg[b])[:20]]))}
             for a, b in combinations(models, 2)]
    pd.DataFrame(cross).to_csv(out_dir / "cross_model_agreement.csv", index=False)
    top = rk[rk.rank_mean <= 20].sort_values(["model", "rank_mean"])
    print("\ntop-20 features:\n", top.round(1).to_string(index=False))
    if cross:
        print("\ncross-model:\n", pd.DataFrame(cross).round(3).to_string(index=False))


# --------------------------------------------------------------------------- main

def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    if args.smoke_test:
        args.n_background, args.n_explain, args.nsamples = 10, 20, 300
        args.output_dir = Path("Results/shap_smoke")
        args.overwrite = True
    import shap
    import matplotlib  # noqa: F401  fail now, not after a long explanation
    print(f"shap {shap.__version__}; models {args.models}; seeds {args.seeds}")
    features, background, explained, groups = load_sets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(background.data, columns=features).assign(weight=background.weights) \
        .to_csv(args.output_dir / "background_centroids.csv", index=False)
    explained.to_csv(args.output_dir / "explained_domains.csv", index=False)
    print(f"correlated groups (|r| >= {args.corr_threshold}): "
          f"{[g for g in groups if len(g) > 1]}")

    all_imp, runs = {}, []
    for model_name in args.models:
        for seed in args.seeds:
            if not (args.results_dir / model_name / f"seed_{seed}").exists():
                print(f"  skip {model_name} seed {seed}: no trained model")
                continue
            out = args.output_dir / model_name / f"seed_{seed}"
            done = out / "shap_values.npy"
            if done.exists() and not args.overwrite:
                values = np.load(done)
                info = json.loads((out / "run.json").read_text())
                print(f"  {model_name} seed {seed}: loaded existing")
            else:
                out.mkdir(parents=True, exist_ok=True)
                values, base, proba, info = explain_one(model_name, seed, features,
                                                        background, explained, args)
                np.save(done, values)
                pd.DataFrame({"Name": explained["Name"], "Label": explained["Label"],
                              "Family": explained["Family"], "p_dga": proba}
                             ).to_csv(out / "predictions.csv", index=False)
                importance_table(values, features).to_csv(out / "importance.csv", index=False)
                consolidated_table(values, features, groups) \
                    .to_csv(out / "importance_consolidated.csv", index=False)
                (out / "run.json").write_text(json.dumps(info, indent=2))
            # Plots are drawn after the values are saved, and redrawn on resume if
            # missing, so a plotting failure never costs a finished explanation.
            if not (out / "summary.png").exists():
                plot_run(values, explained, features, out, f"{model_name} (seed {seed})")
            all_imp.setdefault(model_name, {})[seed] = importance_table(values, features)
            runs.append({"model": model_name, "seed": seed, **info})

    summarise(all_imp, features, groups, args.output_dir / "summary")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(), "shap": shap.__version__,
        "args": {k: str(v) for k, v in vars(args).items()},
        "background": f"{args.n_background} K-means centroids of the training set",
        "runs": runs}, indent=2))
    print(f"\nwritten to {args.output_dir}")


if __name__ == "__main__":
    main()
