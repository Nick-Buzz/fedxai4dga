"""Figures and tables for the near real-time inference evaluation.

    python -m Scripts.realtime.report Results/realtime/<run>

Reads the CSVs written by ``run_benchmark`` and writes, into ``<run>/report``:

  tables     summary.md, table_models.tex, table_stream.tex, and the merged
             CSVs every figure is drawn from (the figures' table view)
  figures    fig_*.pdf and fig_*.png

Colour follows the model, never its rank, and is the same in every figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.realtime.backends import MODEL_LABELS  # noqa: E402

MODEL_ORDER = ["exat_mlp", "single_head", "no_attention", "plain_mlp", "xgboost"]
COLORS = {  # reference categorical palette, fixed per model
    "exat_mlp": "#2a78d6", "plain_mlp": "#eb6834", "xgboost": "#1baf7a",
    "no_attention": "#eda100", "single_head": "#e87ba4",
}
BACKEND_ORDER = ["keras-predict", "tf-eager", "tf-function", "tf-xla", "onnxruntime", "xgb-inplace"]
BACKEND_COLORS = {"keras-predict": "#2a78d6", "tf-eager": "#eb6834", "tf-function": "#1baf7a",
                  "tf-xla": "#eda100", "onnxruntime": "#e87ba4", "xgb-inplace": "#008300"}
POLICY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERVING = {"cpu": {"xgboost": "xgb-inplace", "_keras": "onnxruntime"},
           "gpu": {"xgboost": "xgb-inplace", "_keras": "tf-xla"}}


def serving_backend(model: str, device: str) -> str:
    return SERVING[device]["xgboost" if model == "xgboost" else "_keras"]


def load(run: Path, pattern: str) -> pd.DataFrame:
    parts = [pd.read_csv(p).assign(source=p.name.split(".")[0]) for p in sorted(run.glob(pattern))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ------------------------------------------------------------------- styling

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 8,
        "axes.axisbelow": True,
    })
    return plt


def line(ax, x, y, color, label=None, marker="o", **kw):
    ax.plot(x, y, color=color, label=label, marker=marker, markeredgecolor="white",
            markeredgewidth=1.2, markersize=5.5, **kw)


def save(fig, out: Path, name: str):
    fig.tight_layout()
    if fig._suptitle is not None:            # keep the figure title clear of panel titles
        fig._suptitle.set_y(1.04)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def fmt_rate(x):
    return f"{x / 1000:.0f}k" if x >= 10_000 else (f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}")


# ------------------------------------------------------------------- figures

def fig_latency_throughput(lat, out, plt):
    """Model-only: median latency per call and sustained throughput vs batch size."""
    serving = lat[lat.apply(lambda r: r.backend == serving_backend(r.model, r.device), axis=1)]
    devices = [d for d in ("cpu", "gpu") if d in serving.device.unique()]
    full = serving[(serving.device == "gpu") | (serving.threads == serving.threads.max())]
    fig, axes = plt.subplots(2, len(devices), figsize=(3.6 * len(devices), 5.6),
                             sharex=True, sharey="row", squeeze=False)
    for j, device in enumerate(devices):
        sub = full[full.device == device]
        for model in MODEL_ORDER:
            m = sub[sub.model == model].sort_values("batch_size")
            if m.empty:
                continue
            line(axes[0, j], m.batch_size, m.latency_ms_p50, COLORS[model], MODEL_LABELS[model])
            line(axes[1, j], m.batch_size, m.throughput_per_s, COLORS[model], MODEL_LABELS[model])
        backend = "ONNX Runtime / XGBoost" if device == "cpu" else "XLA / XGBoost-CUDA"
        threads = int(sub.threads.max()) if len(sub) else 0
        axes[0, j].set_title(f"{device.upper()} ({backend}" + (f", {threads} threads)" if device == "cpu" else ")"))
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    axes[0, 0].set_ylabel("latency per call, median (ms)")
    axes[1, 0].set_ylabel("throughput (domains/s)")
    for ax in axes[1]:
        ax.set_xlabel("batch size (domains per call)")
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = [MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in handles]
    fig.legend([handles[k] for k in order], order, loc="lower center", ncol=len(order),
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Model inference only (features already extracted)", fontsize=10, color=INK)
    save(fig, out, "fig_model_latency_throughput")
    full.to_csv(out / "data_model_latency_throughput.csv", index=False)


def fig_backends(lat, out, plt):
    """Single-query latency (batch 1) per serving backend: framework overhead."""
    sub = lat[(lat.batch_size == 1) & ((lat.device == "gpu") | (lat.threads == lat.threads.max()))]
    if sub.empty:
        return
    devices = [d for d in ("cpu", "gpu") if d in sub.device.unique()]
    fig, axes = plt.subplots(1, len(devices), figsize=(3.8 * len(devices), 3.4), sharey=True, squeeze=False)
    models = [m for m in MODEL_ORDER if m in sub.model.unique()]
    for j, device in enumerate(devices):
        ax = axes[0, j]
        d = sub[sub.device == device]
        backends = [b for b in BACKEND_ORDER if b in d.backend.unique()]
        width = 0.8 / max(1, len(backends))
        for k, backend in enumerate(backends):
            vals = [d[(d.model == m) & (d.backend == backend)].latency_ms_p50 for m in models]
            ys = [v.iloc[0] if len(v) else np.nan for v in vals]
            ax.barh(np.arange(len(models)) + (k - (len(backends) - 1) / 2) * width, ys,
                    height=width * 0.85, color=BACKEND_COLORS[backend], label=backend)
        ax.set_xscale("log")
        ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models])
        ax.invert_yaxis()
        ax.set_xlabel("latency, one domain, median (ms, log)")
        ax.set_title(device.upper())
        ax.grid(axis="y", visible=False)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = [b for b in BACKEND_ORDER if b in handles]
    fig.legend([handles[b] for b in order], order, loc="lower center", ncol=len(order),
               fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Serving path overhead at batch size 1", fontsize=10)
    save(fig, out, "fig_backend_overhead")
    sub.to_csv(out / "data_backend_overhead.csv", index=False)


def fig_threads(lat, out, plt):
    """CPU budget: batch-1 latency and throughput vs number of CPU threads."""
    cpu = lat[(lat.device == "cpu") & lat.apply(lambda r: r.backend == serving_backend(r.model, "cpu"), axis=1)]
    if cpu.threads.nunique() < 2:
        return
    big = cpu[cpu.batch_size == 1024] if (cpu.batch_size == 1024).any() else cpu[cpu.batch_size == cpu.batch_size.max()]
    one = cpu[cpu.batch_size == 1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for model in MODEL_ORDER:
        a = one[one.model == model].sort_values("threads")
        b = big[big.model == model].sort_values("threads")
        if a.empty:
            continue
        line(axes[0], a.threads, a.latency_ms_p50, COLORS[model], MODEL_LABELS[model])
        line(axes[1], b.threads, b.throughput_per_s, COLORS[model], MODEL_LABELS[model])
    axes[0].set_ylabel("latency, batch 1, median (ms)")
    axes[1].set_ylabel(f"throughput at batch {int(big.batch_size.iloc[0])} (domains/s)")
    for ax in axes:
        ax.set_xlabel("CPU threads (pinned)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(cpu.threads.unique()), [str(t) for t in sorted(cpu.threads.unique())])
    axes[1].set_yscale("log")
    axes[0].legend()
    fig.suptitle("CPU budget scaling (model inference only)", fontsize=10)
    save(fig, out, "fig_cpu_threads")
    cpu.to_csv(out / "data_cpu_threads.csv", index=False)


def fig_features(run, out, plt):
    stages_file, scaling_file = run / "featurizer_stages.csv", run / "featurizer_scaling.csv"
    if not stages_file.exists():
        return
    stages = pd.read_csv(stages_file).set_index("stage")
    parts = stages.drop(index="total")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
    order = parts.mean_us.sort_values().index
    axes[0].barh(range(len(order)), parts.loc[order, "mean_us"], color=COLORS["exat_mlp"], height=0.6)
    for i, s in enumerate(order):
        axes[0].text(parts.loc[s, "mean_us"], i, f"  {parts.loc[s, 'mean_us']:.1f} µs "
                     f"({parts.loc[s, 'share_of_total']:.0%})", va="center", fontsize=7.5, color=INK2)
    axes[0].set_yticks(range(len(order)), [s.replace("_", " ") for s in order])
    axes[0].set_xlabel("mean time per domain (µs), one thread")
    axes[0].set_xlim(0, parts.mean_us.max() * 1.55)
    axes[0].grid(axis="y", visible=False)
    axes[0].set_title(f"Stages (total {stages.loc['total', 'mean_us']:.0f} µs/domain)")
    if scaling_file.exists():
        sc = pd.read_csv(scaling_file)
        if "variant" not in sc:
            sc["variant"] = "reference"
        for variant, color in (("reference", COLORS["exat_mlp"]), ("single_segmentation", COLORS["plain_mlp"])):
            v = sc[sc.variant == variant].sort_values("processes")
            if v.empty:
                continue
            label = "as built (2× word segmentation)" if variant == "reference" else "1× segmentation, same output"
            line(axes[1], v.processes, v.throughput_per_s, color, label)
        ref = sc[sc.variant == "reference"].sort_values("processes")
        ideal = ref.throughput_per_s.iloc[0] * ref.processes / ref.processes.iloc[0]
        axes[1].plot(ref.processes, ideal, color=INK2, linewidth=1, linestyle=(0, (1, 2)), label="linear")
        axes[1].set_xlabel("worker processes")
        axes[1].set_ylabel("domains/s")
        axes[1].set_title("Parallel feature extraction")
        axes[1].legend()
    fig.suptitle("Raw domain name → 49 scaled features", fontsize=10)
    save(fig, out, "fig_feature_extraction")


def fig_pipeline(pipe, out, plt):
    if pipe.empty:
        return
    """Raw name -> verdict, µs per domain vs batch size; the dotted line is
    feature extraction alone, the floor no model choice can go below."""
    pipe = pipe[pipe.apply(lambda r: r.backend == serving_backend(r.model, r.device), axis=1)]
    devices = [d for d in ("cpu", "gpu") if d in pipe.device.unique()]
    models = [m for m in MODEL_ORDER if m in pipe.model.unique()]
    fig, axes = plt.subplots(1, len(devices), figsize=(3.7 * len(devices), 3.2),
                             sharey=True, squeeze=False)
    for j, device in enumerate(devices):
        ax = axes[0, j]
        d = pipe[pipe.device == device]
        for model in models:
            m = d[d.model == model].sort_values("batch_size")
            if m.empty:
                continue
            total = (m.featurize_ms_median + m.inference_ms_median) / m.batch_size * 1e3
            line(ax, m.batch_size, total, COLORS[model], MODEL_LABELS[model])
        floor = (d.featurize_ms_median / d.batch_size * 1e3).groupby(d.batch_size).median()
        ax.plot(floor.index, floor.values, color=INK2, linewidth=1.2, linestyle=(0, (1, 2)),
                label="feature extraction alone")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_title(f"{device.upper()} ({'ONNX Runtime' if device == 'cpu' else 'XLA'} / XGBoost)")
    axes[0, 0].set_ylabel("µs per domain, median (raw name → verdict)")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("End-to-end cost per domain, one process", fontsize=10)
    save(fig, out, "fig_pipeline_breakdown")
    pipe.to_csv(out / "data_pipeline.csv", index=False)


def fig_stream_policy(stream, out, plt):
    pol = stream[stream.experiment == "policy"]
    if pol.empty:
        return
    configs = [(m, d) for m in MODEL_ORDER for d in ("cpu", "gpu")
               if len(pol[(pol.model == m) & (pol.device == d)])]
    fig, axes = plt.subplots(2, len(configs), figsize=(2.9 * len(configs), 5.4),
                             sharey="row", squeeze=False)
    policies = pol[["max_batch", "max_wait_ms"]].drop_duplicates().sort_values("max_batch").values.tolist()
    for j, (model, device) in enumerate(configs):
        d = pol[(pol.model == model) & (pol.device == device)]
        for k, (b, w) in enumerate(policies):
            p = d[(d.max_batch == b) & (d.max_wait_ms == w)].sort_values("offered_rate")
            ok = p[p.stable]
            if ok.empty:
                continue
            lab = "no batching" if b == 1 else f"≤{int(b)} / {w:g} ms"
            line(axes[0, j], ok.offered_rate, ok.latency_ms_p99, POLICY_COLORS[k], lab)
            line(axes[1, j], ok.offered_rate, ok.latency_ms_p50, POLICY_COLORS[k], lab)
        axes[0, j].set_title(f"{MODEL_LABELS[model]} ({device.upper()})")
        axes[1, j].set_xlabel("offered load (queries/s)")
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            for slo in (1, 10, 100):
                ax.axhline(slo, color=INK2, linewidth=0.8, linestyle=(0, (1, 2)))
    axes[0, 0].set_ylabel("end-to-end latency p99 (ms)")
    axes[1, 0].set_ylabel("end-to-end latency p50 (ms)")
    axes[0, 0].legend(title="max batch / max wait", fontsize=7, title_fontsize=7)
    fig.suptitle("Live traffic, one core: latency vs offered load, stable points only "
                 "(dotted: 1, 10, 100 ms)", fontsize=10)
    save(fig, out, "fig_stream_policies")


def capacity_table(stream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["experiment", "model", "backend", "device", "shards", "threads_per_shard", "cpus_total",
            "max_batch", "max_wait_ms"]
    for key, g in stream.groupby(keys):
        g = g.sort_values("offered_rate")
        stable = g[g.stable]
        rec = dict(zip(keys, key))
        rec["capacity_qps"] = float(stable.offered_rate.max()) if len(stable) else 0.0
        for slo in (1, 10, 100):
            ok = stable[stable.latency_ms_p99 <= slo]
            rec[f"capacity_p99_le_{slo}ms"] = float(ok.offered_rate.max()) if len(ok) else 0.0
        if len(stable):
            top = stable.iloc[-1]
            rec.update({"p50_ms_at_capacity": top.latency_ms_p50, "p99_ms_at_capacity": top.latency_ms_p99,
                        "utilization_at_capacity": top.utilization})
            low = stable.iloc[0]
            rec.update({"lowest_rate": low.offered_rate, "p50_ms_low_load": low.latency_ms_p50,
                        "p99_ms_low_load": low.latency_ms_p99})
        rows.append(rec)
    return pd.DataFrame(rows)


def fig_stream_scaling(cap, out, plt):
    sc = cap[(cap.device == "cpu") & (cap.experiment == "scaling")]
    if sc.shards.nunique() < 2:
        return
    policy = sc[sc.shards > 1][["max_batch", "max_wait_ms"]].drop_duplicates()
    if policy.empty:
        return
    b, w = policy.iloc[0]
    sc = sc[(sc.max_batch == b) & (sc.max_wait_ms == w)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for model in MODEL_ORDER:
        m = sc[sc.model == model].sort_values("shards")
        if m.empty:
            continue
        line(axes[0], m.shards, m.capacity_qps, COLORS[model], MODEL_LABELS[model])
        line(axes[1], m.shards, m.capacity_p99_le_10ms, COLORS[model], MODEL_LABELS[model])
    axes[0].set_title("sustainable load")
    axes[1].set_title("load with p99 ≤ 10 ms")
    for ax in axes:
        ax.set_xlabel("pipeline processes (1 pinned core each)")
        ax.set_ylabel("queries/s")
        ax.set_ylim(bottom=0)
        ax.set_xticks(sorted(sc.shards.unique()))
    axes[0].legend()
    fig.suptitle(f"Horizontal scaling, raw name → verdict (max batch {int(b)}, max wait {w:g} ms)", fontsize=10)
    save(fig, out, "fig_stream_scaling")


def fig_stream_cdf(run, stream, out, plt):
    hist = load(run, "stream_hist_policy_*.csv.gz")
    if hist.empty:
        return
    from Scripts.realtime.stream import HIST_EDGES_LOG10_MS
    edges = np.logspace(*HIST_EDGES_LOG10_MS)
    configs = [(m, d) for m in MODEL_ORDER for d in ("cpu", "gpu")
               if len(hist[(hist.model == m) & (hist.device == d)])]
    fig, axes = plt.subplots(1, len(configs), figsize=(2.9 * len(configs), 2.9), sharey=True, squeeze=False)
    target = 1000.0
    for j, (model, device) in enumerate(configs):
        h = hist[(hist.model == model) & (hist.device == device) & (hist.offered_rate == target)]
        s = stream[(stream.model == model) & (stream.device == device) & (stream.experiment == "policy")
                   & (stream.offered_rate == target)]
        policies = h[["max_batch", "max_wait_ms"]].drop_duplicates().sort_values("max_batch").values.tolist()
        for k, (b, w) in enumerate(policies):
            ok = s[(s.max_batch == b) & (s.max_wait_ms == w)]
            if ok.empty or not bool(ok.stable.iloc[0]):
                continue
            counts = np.array(h[(h.max_batch == b) & (h.max_wait_ms == w)].counts.iloc[0].split(), dtype=float)
            cdf = np.cumsum(counts) / counts.sum()
            lab = "no batching" if b == 1 else f"≤{int(b)} / {w:g} ms"
            axes[0, j].plot(edges[1:], cdf, color=POLICY_COLORS[k], label=lab)
        axes[0, j].set_xscale("log")
        axes[0, j].set_title(f"{MODEL_LABELS[model]} ({device.upper()})")
        axes[0, j].set_xlabel("end-to-end latency (ms)")
    axes[0, 0].set_ylabel("fraction of queries")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Latency distribution at {target:,.0f} queries/s, one core", fontsize=10)
    save(fig, out, "fig_stream_cdf")


def fig_tradeoff(cap, out, plt):
    agg_file = PROJECT_ROOT / "Results/full_run/aggregate_metrics.csv"
    if cap.empty or not agg_file.exists():
        return
    agg = agg_file.exists() and pd.read_csv(agg_file)
    one = cap[(cap.experiment == "policy") & (cap.device == "cpu")]
    best = one.groupby("model").capacity_qps.max()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for model in MODEL_ORDER:
        f1 = agg[(agg.model == model) & (agg.metric == "f1")]
        if model not in best.index or f1.empty:
            continue
        x, y, e = best[model], f1["mean"].iloc[0] * 100, f1["std"].iloc[0] * 100
        ax.errorbar(x, y, yerr=e, fmt="o", color=COLORS[model], markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, capsize=3)
        ax.annotate(MODEL_LABELS[model], (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color=INK2)
    ax.set_xlabel("sustainable queries/s on one core (raw name → verdict)")
    ax.set_ylabel("test F1 (%), mean ± std over 5 seeds")
    ax.set_title("Accuracy vs serving cost", fontsize=10)
    save(fig, out, "fig_accuracy_vs_capacity")


def fig_idle_gap(run, out, plt):
    """Single-query latency after an idle gap: why light load is slower than moderate load."""
    path = run / "idle_gap_probe.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for model in MODEL_ORDER:
        m = d[d.model == model].sort_values("idle_gap_ms")
        if m.empty:
            continue
        x = m.idle_gap_ms.replace(0, 0.1)          # 0 ms plotted at 0.1 on the log axis
        line(ax, x, m.latency_ms_p50, COLORS[model], MODEL_LABELS[model])
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 5, 10, 50], ["0", "1", "5", "10", "50"])
    ax.set_xlabel("idle time before each query (ms)")
    ax.set_ylabel("latency, raw name → verdict, p50 (ms)")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title("Cold-core penalty, one pinned core", fontsize=10)
    save(fig, out, "fig_idle_gap")


# -------------------------------------------------------------------- tables

def model_table(lat, cold, fid, full, pipe, cap) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        r = {"model": MODEL_LABELS[model]}
        for device in ("cpu", "gpu"):
            be = serving_backend(model, device)
            d = lat[(lat.model == model) & (lat.device == device) & (lat.backend == be)]
            if device == "cpu":
                d = d[d.threads == d.threads.max()] if len(d) else d
            one = d[d.batch_size == 1]
            if len(one):
                r[f"{device}_b1_p50_ms"] = one.latency_ms_p50.iloc[0]
                r[f"{device}_b1_p99_ms"] = one.latency_ms_p99.iloc[0]
            if len(d):
                r[f"{device}_peak_per_s"] = d.throughput_per_s.max()
                r[f"{device}_peak_batch"] = int(d.loc[d.throughput_per_s.idxmax(), "batch_size"])
        c = cold[(cold.model == model) & (cold.device == "cpu")
                 & (cold.backend == serving_backend(model, "cpu"))] if len(cold) else cold
        if len(c):
            c = c.iloc[0]
            r["size_mb"] = c.artifact_bytes / 2**20
            r["load_s"] = c.load_s
            r["first_call_ms"] = c.first_call_ms
            if "parameters" in c and pd.notna(c.get("parameters")):
                r["parameters"] = int(c.parameters)
        k = cold[(cold.model == model) & (cold.backend.isin(["tf-function"]))] if len(cold) else cold
        if len(k) and "parameters" in k:
            r["parameters"] = int(k.parameters.iloc[0])
        if len(fid):
            f = fid[fid.model == model]
            r["max_prob_diff"] = f.max_abs_diff.max() if len(f) else np.nan
            r["verdict_agreement_min"] = f.verdict_agreement.min() if len(f) else np.nan
        if len(full):
            f = full[(full.model == model) & (full.device == "cpu")]
            if len(f):
                r["accuracy_served"] = f.accuracy.iloc[0]
                r["accuracy_reported"] = f.reported_accuracy.iloc[0]
        if len(pipe):
            p = pipe[(pipe.model == model) & (pipe.device == "cpu") & (pipe.batch_size == 1)]
            if len(p):
                r["e2e_b1_p50_ms"] = p.latency_ms_p50.iloc[0]
                r["e2e_b1_p99_ms"] = p.latency_ms_p99.iloc[0]
        if len(cap):
            c1 = cap[(cap.model == model) & (cap.device == "cpu") & (cap.experiment == "policy")]
            if len(c1):
                r["stream_1core_qps"] = c1.capacity_qps.max()
            cn = cap[(cap.model == model) & (cap.device == "cpu") & (cap.experiment == "scaling")]
            if len(cn):
                best = cn.loc[cn.capacity_qps.idxmax()]
                r["stream_best_qps"] = best.capacity_qps
                r["stream_best_shards"] = int(best.shards)
        rows.append(r)
    return pd.DataFrame(rows)


def write_latex(df: pd.DataFrame, path: Path, caption: str, label: str):
    cols = " ".join(["l"] + ["r"] * (df.shape[1] - 1))
    lines = [r"\begin{table}[t]", r"\centering\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(df.columns) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines).replace("%", r"\%").replace("_", r"\_"), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run
    out = run / "report"
    out.mkdir(exist_ok=True)
    plt = setup_matplotlib()

    lat = load(run, "latency_*.csv")
    cold = load(run, "coldstart_*.csv")
    fid = load(run, "fidelity_*.csv")
    full = load(run, "fulltest_*.csv")
    pipe = load(run, "pipeline_*.csv")
    stream = load(run, "stream_*.csv")
    if len(stream):
        stream["experiment"] = np.where(stream.source.str.startswith("stream_policy"), "policy", "scaling")
    cap = capacity_table(stream) if len(stream) else pd.DataFrame()
    env = json.loads((run / "environment.json").read_text()) if (run / "environment.json").exists() else {}

    for name, df in [("latency", lat), ("coldstart", cold), ("fidelity", fid), ("fulltest", full),
                     ("pipeline", pipe), ("stream", stream), ("stream_capacity", cap)]:
        if len(df):
            df.to_csv(out / f"all_{name}.csv", index=False)

    if len(lat):
        fig_latency_throughput(lat, out, plt)
        fig_backends(lat, out, plt)
        fig_threads(lat, out, plt)
    fig_features(run, out, plt)
    fig_idle_gap(run, out, plt)
    fig_pipeline(pipe, out, plt)
    if len(stream):
        fig_stream_policy(stream, out, plt)
        fig_stream_scaling(cap, out, plt)
        fig_stream_cdf(run, stream, out, plt)
        fig_tradeoff(cap, out, plt)

    if lat.empty:
        print("no latency results; skipping the per-model table")
        return
    table = model_table(lat, cold, fid, full, pipe, cap)
    table.to_csv(out / "table_models.csv", index=False)
    show = table.copy()
    for c in show.columns:
        if c == "model":
            continue
        if show[c].dtype.kind == "f":
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else
                                  (f"{v:,.0f}" if abs(v) >= 100 else (f"{v:.3g}" if abs(v) >= 1e-3 else f"{v:.1e}")))
    write_latex(show[[c for c in ["model", "parameters", "size_mb", "cpu_b1_p50_ms", "cpu_b1_p99_ms",
                                  "cpu_peak_per_s", "gpu_b1_p50_ms", "gpu_peak_per_s", "e2e_b1_p50_ms",
                                  "stream_1core_qps", "stream_best_qps"] if c in show]],
                out / "table_models.tex",
                "Inference cost per model. Latency in ms, throughput in domains/s; "
                "e2e includes feature extraction from the raw name; stream capacity is the highest "
                "Poisson load served without backlog.", "tab:inference")
    if len(cap):
        cap_show = cap.sort_values(["model", "device", "shards", "max_batch"])
        cap_show.to_csv(out / "table_stream_capacity.csv", index=False)

    md = ["# Near real-time inference evaluation", ""]
    if env:
        md += [f"- Machine: {env.get('cpu_model')} ({env.get('cpus_available')} logical CPUs), "
               f"{env.get('memory_gb')} GB RAM visible, GPU {env.get('gpu')}",
               f"- Software: TensorFlow {env.get('tensorflow')}, ONNX Runtime {env.get('onnxruntime')}, "
               f"XGBoost {env.get('xgboost')}, Python {env.get('python')}, WSL2={env.get('wsl')}",
               f"- Commit {env.get('git_commit', '')[:10]} ({env.get('git_branch')}), created {env.get('created')}",
               "- Classification only; no SHAP.", ""]
    md += ["## Per model", "", show.to_markdown(index=False), ""]
    if len(cap):
        md += ["## Streaming capacity", "",
               cap_show.round(3).to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
