"""Summaries, statistics and plots from a benchmark run (only saved, measured values).

Produces in ``<run>/report/``:
    summary_by_method.csv     mean/std/n per method x scale x degradation
    summary_by_category.csv   same, per image category (categories overlap)
    roi_summary.csv           ROI-level means
    paired_vs_bicubic.csv     per-image paired differences vs bicubic with bootstrap 95% CI + win rate
    fig_*.png                 plots (see below)
    results.md                tables + not-run list + interpretation template
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Reference palette (dataviz skill, light mode). Methods are coloured by *family*
# (identity of the approach), never by rank; first three slots validate all-pairs.
FAMILY_COLOURS = {"classical": "#2a78d6", "ours": "#eb6834", "pretrained": "#1baf7a"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["psnr_y", "ssim_y", "psnr_rgb", "edge_psnr", "overshoot", "lpips", "runtime_s"]
HIGHER_BETTER = {"psnr_y": True, "ssim_y": True, "psnr_rgb": True, "edge_psnr": True,
                 "overshoot": False, "lpips": False, "runtime_s": False}


def family(label: str) -> str:
    l = label.lower()
    if l in ("nearest", "bilinear", "bicubic") or l.startswith("bicubic"):
        return "classical"
    if "esrgan" in l:
        return "pretrained"
    return "ours"


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = [m for m in METRICS if m in df and df[m].notna().any()]
    g = df.groupby(keys)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n"))
    return out.reset_index()


def explode_categories(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(category=df["categories"].fillna("").str.split(";")).explode("category")
    d["category"] = d["category"].replace("", "uncategorised")
    return d


def paired_vs_reference(df: pd.DataFrame, ref: str = "Bicubic") -> pd.DataFrame:
    rows = []
    for (s, d), grp in df.groupby(["scale", "degradation"]):
        base = grp[grp.method == ref].set_index("image_id")
        if base.empty:
            continue
        for m, mg in grp.groupby("method"):
            if m == ref:
                continue
            j = mg.set_index("image_id").join(base, rsuffix="_ref", how="inner")
            for metric in ("psnr_y", "ssim_y", "lpips"):
                if metric not in j or j[metric].isna().all():
                    continue
                diff = (j[metric] - j[f"{metric}_ref"]).to_numpy()
                lo, hi = bootstrap_ci(diff)
                better = diff > 0 if HIGHER_BETTER[metric] else diff < 0
                rows.append({"scale": s, "degradation": d, "method": m, "metric": metric, "reference": ref,
                             "mean_diff": diff.mean(), "ci95_low": lo, "ci95_high": hi,
                             "win_rate": better.mean(), "n_images": len(diff)})
    return pd.DataFrame(rows)


def plot_method_bars(summ: pd.DataFrame, metric: str, out: Path, ylabel: str) -> None:
    import matplotlib.pyplot as plt
    combos = summ[["scale", "degradation"]].drop_duplicates().values.tolist()
    if f"{metric}_mean" not in summ:
        return
    fig, axes = plt.subplots(1, len(combos), figsize=(4.2 * len(combos), 3.6), squeeze=False)
    for ax, (s, d) in zip(axes[0], combos):
        sub = summ[(summ.scale == s) & (summ.degradation == d)]
        x = np.arange(len(sub))
        cols = [FAMILY_COLOURS[family(m)] for m in sub.method]
        ax.bar(x, sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"], color=cols, width=0.6,
               error_kw={"elinewidth": 0.8, "ecolor": MUTED, "capsize": 2}, edgecolor="white", linewidth=1)
        lo = np.nanmin(sub[f"{metric}_mean"] - sub[f"{metric}_std"].fillna(0))
        hi = np.nanmax(sub[f"{metric}_mean"] + sub[f"{metric}_std"].fillna(0))
        if metric.startswith("psnr") or metric == "edge_psnr":
            ax.set_ylim(lo - 0.05 * (hi - lo) - 0.5, hi + 0.05 * (hi - lo))
        ax.set_xticks(x, sub.method, rotation=35, ha="right")
        ax.set_title(f"x{s}, {d} LR", fontsize=10, color=TEXT)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        _style(ax)
    names = {"classical": "classical interpolation", "ours": "our CNNs", "pretrained": "pretrained (Real-ESRGAN)"}
    present = [f for f in FAMILY_COLOURS if any(family(m) == f for m in summ.method)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOURS[f]) for f in present]
    fig.legend(handles, [names[f] for f in present], loc="upper center", ncol=len(present), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_perception_distortion(summ: pd.DataFrame, out: Path) -> None:
    """PSNR (distortion) vs LPIPS (perception) - Blau & Michaeli's trade-off plane."""
    import matplotlib.pyplot as plt
    ycol = "lpips_mean" if "lpips_mean" in summ and summ["lpips_mean"].notna().any() else "ssim_y_mean"
    combos = summ[["scale", "degradation"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(combos), figsize=(4.2 * len(combos), 3.6), squeeze=False)
    for ax, (s, d) in zip(axes[0], combos):
        sub = summ[(summ.scale == s) & (summ.degradation == d)]
        for _, r in sub.iterrows():
            ax.scatter(r.psnr_y_mean, r[ycol], s=60, color=FAMILY_COLOURS[family(r.method)],
                       edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(r.method, (r.psnr_y_mean, r[ycol]), xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color=TEXT)
        ax.set_xlabel("PSNR-Y (dB)  - higher = more faithful pixels", fontsize=8, color=MUTED)
        ax.set_ylabel("LPIPS - lower = perceptually closer" if ycol == "lpips_mean" else "SSIM-Y",
                      fontsize=8, color=MUTED)
        ax.set_title(f"x{s}, {d} LR", fontsize=10, color=TEXT)
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_category_delta(paired_cat: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    if paired_cat.empty:
        return
    piv = paired_cat.pivot_table(index="category", columns="method", values="mean_diff")
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(piv.columns), 0.5 + 0.45 * len(piv)))
    lim = np.nanmax(np.abs(piv.values)) or 1
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * lim else TEXT)
    fig.colorbar(im, ax=ax, label="PSNR-Y gain over bicubic (dB)")
    ax.set_title("Category-wise gain over bicubic", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build_report(run_dir: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    run = Path(run_dir)
    rep = run / "report"
    rep.mkdir(exist_ok=True)
    df = pd.read_csv(run / "metrics_per_image.csv")
    status = pd.read_csv(run / "status.csv") if (run / "status.csv").exists() else pd.DataFrame()
    keys = ["scale", "degradation", "method"]
    order = list(dict.fromkeys(df.method))
    df["method"] = pd.Categorical(df.method, order, ordered=True)
    summ = summarise(df, keys)
    summ.to_csv(rep / "summary_by_method.csv", index=False)
    cat = explode_categories(df)
    summ_cat = summarise(cat, ["category"] + keys)
    summ_cat.to_csv(rep / "summary_by_category.csv", index=False)
    df["method"] = df.method.astype(str)
    paired = paired_vs_reference(df)
    paired.to_csv(rep / "paired_vs_bicubic.csv", index=False)
    pc = []
    for c, g in explode_categories(df).groupby("category"):
        p = paired_vs_reference(g)
        if not p.empty:
            pc.append(p.assign(category=c))
    paired_cat = pd.concat(pc) if pc else pd.DataFrame()
    if not paired_cat.empty:
        paired_cat.to_csv(rep / "paired_vs_bicubic_by_category.csv", index=False)

    summ_plot = summ.copy()
    summ_plot["method"] = summ_plot.method.astype(str)
    plot_method_bars(summ_plot, "psnr_y", rep / "fig_psnr_by_method.png", "PSNR-Y (dB), mean ± sd")
    plot_method_bars(summ_plot, "ssim_y", rep / "fig_ssim_by_method.png", "SSIM-Y, mean ± sd")
    plot_perception_distortion(summ_plot, rep / "fig_perception_distortion.png")
    if not paired_cat.empty:
        for (s, d), g in paired_cat[paired_cat.metric == "psnr_y"].groupby(["scale", "degradation"]):
            plot_category_delta(g, rep / f"fig_category_gain_x{s}_{d}.png")

    roi_md = ""
    if (run / "roi_metrics.csv").exists():
        roi = pd.read_csv(run / "roi_metrics.csv")
        if not roi.empty:
            rs = roi.groupby(["scale", "degradation", "roi", "method"], sort=False)[
                ["psnr_y", "ssim_y", "edge_psnr", "overshoot"]].mean().reset_index()
            rs = rs.join(roi.groupby(["scale", "degradation", "roi", "method"], sort=False).size()
                         .rename("n").reset_index()["n"])
            rs.to_csv(rep / "roi_summary.csv", index=False)
            roi_md = rs.to_markdown(index=False, floatfmt=".3f")

    fmt = lambda t: t.to_markdown(index=False, floatfmt=".3f") if not t.empty else "_none_"  # noqa: E731
    not_run = status[status.status != "ran"] if not status.empty else status
    cols = [c for c in summ_plot.columns if c.endswith("_mean") or c in keys + ["n"]]
    md = [f"# Benchmark results: `{run.name}`\n",
          "All numbers below are computed from saved outputs of this run "
          f"(`{run}/metrics_per_image.csv`). Conventions: PSNR/SSIM on BT.601 Y, border crop = scale.\n",
          "## Method summary\n", fmt(summ_plot[cols]), "\n",
          "## Paired difference vs bicubic (bootstrap 95% CI, win rate = fraction of images improved)\n",
          fmt(paired), "\n", "## ROI / detail-preservation summary\n", roi_md or "_no ROI data_", "\n",
          "## Methods not run\n", fmt(not_run) if not not_run.empty else "_all configured methods ran_", "\n",
          "## Interpretation template (fill in from the evidence above)\n",
          "- **Evidence (measured):** …\n- **Observations (visual, qualitative):** …\n"
          "- **Limitations:** sample size per category, synthetic degradation vs real LR, "
          "PSNR/SSIM do not prove detail authenticity, LPIPS is a learned proxy.\n"]
    (rep / "results.md").write_text("\n".join(md), encoding="utf-8")
    return rep
