"""Build the results table directly from the evaluation JSONs.

Reads every ``metrics.json`` under a run directory, aggregates across seeds,
and emits LaTeX and Markdown with a provenance file listing every source.  It
refuses to emit a table when models disagree on which seeds they ran, since
the rows would then not be comparable.

    python -m Scripts.make_results_table Results/experiments/<stamp>
    python -m Scripts.make_results_table <dir> --metrics accuracy recall f1 roc_auc
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METRICS = [
    ("accuracy", "Accuracy", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1", "F1", True),
    ("false_positive_rate", "FPR", False),   # lower is better
    ("roc_auc", "ROC-AUC", True),
]

MODEL_LABELS = {
    "exat_mlp": "ExAt-MLP (ours)",
    "no_attention": "Ablation: no attention",
    "single_head": "Ablation: 1 head",
    "plain_mlp": "MLP",
    "xgboost": "XGBoost",
    "autoencoder": "Autoencoder",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--metrics", nargs="+", default=[m[0] for m in DEFAULT_METRICS])
    p.add_argument("--order", nargs="+", default=None,
                   help="model order in the table; default is directory order")
    p.add_argument("--allow-ragged", action="store_true",
                   help="emit even when models ran different seed sets. Only use "
                        "this knowingly.")
    return p.parse_args()


def collect(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("*/seed_*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        dir_model = path.parent.parent.name
        dir_seed = int(path.parent.name.split("_")[1])
        rec_model = record.get("model", dir_model)
        rec_seed = record.get("seed", dir_seed)

        # A results directory that has been moved, renamed or hand-edited cannot
        # be trusted to say which run produced which numbers.
        if (rec_model, rec_seed) != (dir_model, dir_seed):
            raise SystemExit(
                "PROVENANCE MISMATCH. REFUSING TO EMIT TABLE.\n"
                f"    {path.relative_to(run_dir)}\n"
                f"    directory says : {dir_model} / seed {dir_seed}\n"
                f"    file records   : {rec_model} / seed {rec_seed}\n"
                "Results directories must not be moved or renamed after a run. "
                "Re-run instead."
            )

        record["_model"] = rec_model
        record["_seed"] = rec_seed
        record["_source"] = str(path.relative_to(run_dir))
        rows.append(record)
    if not rows:
        raise SystemExit(f"no metrics.json found under {run_dir}")
    return pd.DataFrame(rows)


def check_comparable(frame: pd.DataFrame, allow_ragged: bool) -> None:
    """Require every model to have run the same seeds."""
    seeds_by_model = {m: sorted(g["_seed"].unique()) for m, g in frame.groupby("_model")}
    distinct = {tuple(v) for v in seeds_by_model.values()}
    if len(distinct) > 1:
        report = "\n".join(f"    {m:<16} seeds {v}" for m, v in seeds_by_model.items())
        message = (
            "Models did not run the same seeds, so this is not a like-for-like "
            f"comparison:\n{report}\n"
            "Re-run the missing seeds (--resume skips completed ones), or pass "
            "--allow-ragged and report the discrepancy."
        )
        if not allow_ragged:
            raise SystemExit("REFUSING TO EMIT TABLE.\n" + message)
        print("WARNING: " + message, file=sys.stderr)

    duplicates = frame.duplicated(subset=["_model", "_seed"]).sum()
    if duplicates:
        raise SystemExit(f"{duplicates} duplicate model/seed pairs found; clean the run directory")


def summarise(frame: pd.DataFrame, metrics, order=None) -> pd.DataFrame:
    models = order or list(dict.fromkeys(frame["_model"]))
    out = []
    for model in models:
        group = frame[frame["_model"] == model]
        if group.empty:
            continue
        row = {"model": model, "seeds": len(group)}
        for metric in metrics:
            if metric not in group:
                row[metric] = None
                continue
            row[metric] = group[metric].mean()
            row[metric + "_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


def best_index(summary: pd.DataFrame, metric: str, higher_is_better: bool):
    if metric not in summary or summary[metric].isna().all():
        return None
    return summary[metric].idxmax() if higher_is_better else summary[metric].idxmin()


def to_latex(summary: pd.DataFrame, metrics, provenance: dict) -> str:
    spec = {m[0]: m for m in DEFAULT_METRICS}
    cols = [spec.get(m, (m, m.replace("_", " ").title(), True)) for m in metrics]

    lines = [
        "% Generated by Scripts/make_results_table.py -- do not edit by hand.",
        f"% run        : {provenance['run_dir']}",
        f"% git commit : {provenance['git_commit']}",
        f"% generated  : {provenance['generated_at']}",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Mean $\\pm$ standard deviation over "
        f"{provenance['seeds']} seeds. Precision, recall and F1 are for the "
        "malicious class; ROC-AUC is computed from predicted probabilities.}",
        "\\label{tab:results}",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        "Model & " + " & ".join(c[1] for c in cols) + " \\\\",
        "\\midrule",
    ]
    for i, row in summary.iterrows():
        cells = []
        for key, _, higher in cols:
            value = row.get(key)
            if value is None or pd.isna(value):
                cells.append("--")
                continue
            std = row.get(key + "_std", 0.0) or 0.0
            text = f"{value * 100:.2f} $\\pm$ {std * 100:.2f}"
            if i == best_index(summary, key, higher):
                text = "\\textbf{" + text + "}"
            cells.append(text)
        lines.append(f"{MODEL_LABELS.get(row['model'], row['model'])} & " +
                     " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def to_markdown(summary: pd.DataFrame, metrics) -> str:
    spec = {m[0]: m for m in DEFAULT_METRICS}
    cols = [spec.get(m, (m, m.replace("_", " ").title(), True)) for m in metrics]
    head = "| Model | seeds | " + " | ".join(c[1] for c in cols) + " |"
    rule = "|---|---|" + "---|" * len(cols)
    lines = [head, rule]
    for _, row in summary.iterrows():
        cells = []
        for key, _, _ in cols:
            value = row.get(key)
            cells.append("--" if value is None or pd.isna(value)
                         else f"{value * 100:.2f} ± {(row.get(key + '_std') or 0) * 100:.2f}")
        lines.append(f"| {MODEL_LABELS.get(row['model'], row['model'])} | "
                     f"{int(row['seeds'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    frame = collect(args.run_dir)
    check_comparable(frame, args.allow_ragged)
    summary = summarise(frame, args.metrics, args.order)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=PROJECT_ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"

    provenance = {
        "run_dir": str(args.run_dir),
        "git_commit": commit,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "seeds": int(summary["seeds"].max()),
        "sources": sorted(frame["_source"]),
    }

    (args.run_dir / "results_table.tex").write_text(
        to_latex(summary, args.metrics, provenance), encoding="utf-8")
    (args.run_dir / "results_table.md").write_text(
        to_markdown(summary, args.metrics), encoding="utf-8")
    (args.run_dir / "results_table_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    summary.to_csv(args.run_dir / "results_table.csv", index=False)

    print(to_markdown(summary, args.metrics))
    print(f"\nEvery cell above traces to one of {len(provenance['sources'])} "
          f"metrics.json files listed in results_table_provenance.json")
    print(f"Wrote results_table.tex, .md, .csv and provenance to {args.run_dir}")


if __name__ == "__main__":
    main()
