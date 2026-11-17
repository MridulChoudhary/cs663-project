"""Presentation/report figures built from a finished benchmark run + the trained models.

python scripts/make_figures.py --run outputs/eval/main --out outputs/figures

* Region crops (text, faces, edges) on poster test images: every method is run on the LR
  crop plus a context margin (64 LR px) and the centre is kept, so the pixels match
  whole-image inference for our CNNs (receptive field << margin); for Real-ESRGAN the
  margin is large relative to typical tiling overlaps. Captions show the ROI PSNR-Y taken
  from the benchmark CSV (whole-image inference on Colab), never recomputed numbers.
* Summary charts (method bars by source, perception-distortion plane, ablations, 3x vs 6x,
  runtime) read only the saved CSVs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faithful_sr.inference.upscale import get_method  # noqa: E402
from faithful_sr.models import interpolation as interp  # noqa: E402
from faithful_sr.utils.image_io import load_image, save_image, to_uint8  # noqa: E402

FAMILY = {"classical": "#2a78d6", "ours": "#eb6834", "pretrained": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CROP_METHODS = [("Bicubic", "bicubic", {}),
                ("SRCNN", "cnn", {"checkpoint": "models/srcnn_x3_l1_bicubic.pt"}),
                ("FSRNet-L1", "cnn", {"checkpoint": "models/fsrnet_x3_l1_bicubic.pt"}),
                ("Real-ESRGAN", "realesrgan", {})]


def family(m: str) -> str:
    l = m.lower()
    if l in ("nearest", "bilinear", "bicubic"):
        return "classical"
    return "pretrained" if "esrgan" in l else "ours"


def region_crops(run: Path, out: Path, specs, processed: Path, margin: int = 64) -> None:
    from PIL import Image, ImageDraw, ImageFont
    rois = json.loads((run / "rois_used.json").read_text())
    roi_m = pd.read_csv(run / "roi_metrics.csv")
    fns = {}
    for spec in specs:
        iid, label, deg, scale, methods = spec["id"], spec["roi"], spec.get("deg", "bicubic"), 3, spec.get("methods", CROP_METHODS)
        box = next(r["box"] for r in rois[iid] if r["label"] == label)
        x, y, w, h = box
        if spec.get("sub"):          # zoom into part of a long region: (dx, width) in HR px, multiples of 3
            dx, sw = spec["sub"]
            x, w = x + dx, sw
        hr = load_image(processed / "test" / "HR" / f"{iid}.png").rgb
        lr = load_image(processed / "test" / f"x{scale}" / deg / f"{iid}.png").rgb
        lx0, ly0, lx1, ly1 = x // scale, y // scale, (x + w) // scale, (y + h) // scale
        mx0, my0 = max(0, lx0 - margin), max(0, ly0 - margin)
        mx1, my1 = min(lr.shape[1], lx1 + margin), min(lr.shape[0], ly1 + margin)
        lr_c = lr[my0:my1, mx0:mx1]
        oy, ox = (ly0 - my0) * scale, (lx0 - mx0) * scale
        tiles = [("Input (LR, nearest)", interp.upscale(lr[ly0:ly1, lx0:lx1], scale, "nearest"), None)]
        for name, meth, kw in methods:
            key = (meth, json.dumps(kw, sort_keys=True))
            if key not in fns:
                fns[key] = get_method(meth, **kw)
            sr = fns[key](lr_c, scale)[oy:oy + h, ox:ox + w]
            row = roi_m[(roi_m.image_id == iid) & (roi_m.method == name) & (roi_m.scale == scale)
                        & (roi_m.degradation == deg) & (roi_m.roi == label)]
            tiles.append((name, sr, float(row.psnr_y.iloc[0]) if len(row) else None))
        tiles.append(("Ground truth (HR)", hr[y:y + h, x:x + w], None))
        if spec.get("no_input"):
            tiles = [t for t in tiles if not t[0].startswith("Input")]
        if spec.get("layout") == "grid":
            _grid(tiles, out / f"slide_{iid}_{label}_{deg}.png", spec.get("cols", 2), spec.get("tile_w", 900))
            print("[figures]", f"slide_{iid}_{label}_{deg}.png (grid)")
            continue
        if w / h > 1.6:
            _stack_rows(tiles, out / f"crop_{iid}_{label}_{deg}.png", spec.get("tile_w", 1100))
            print("[figures]", f"crop_{iid}_{label}_{deg}.png (rows)")
            continue
        # compose: equal-height tiles in one row, captions under each
        th = spec.get("tile_h", 260)
        ims = []
        for name, im, p in tiles:
            s = th / im.shape[0]
            r = interp.resize(im, th, max(1, round(im.shape[1] * s)), "nearest" if s >= 1 else "bicubic")
            ims.append((name, to_uint8(r), p))
        pad, cap = 10, 46
        W = sum(i.shape[1] for _, i, _ in ims) + pad * (len(ims) + 1)
        canvas = Image.new("RGB", (W, th + cap + 2 * pad), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        try:
            f1, f2 = ImageFont.truetype("DejaVuSans-Bold.ttf", 17), ImageFont.truetype("DejaVuSans.ttf", 15)
        except OSError:
            f1 = f2 = ImageFont.load_default()
        xx = pad
        for name, im, p in ims:
            canvas.paste(Image.fromarray(im), (xx, pad))
            col = {"classical": FAMILY["classical"], "ours": FAMILY["ours"], "pretrained": FAMILY["pretrained"]}.get(
                family(name), INK) if name not in ("Input (LR, nearest)", "Ground truth (HR)") else INK
            d.rectangle([xx, pad + th, xx + im.shape[1], pad + th + 4], fill=col)
            d.text((xx, pad + th + 8), name, fill=INK, font=f1)
            if p is not None:
                d.text((xx, pad + th + 27), f"region PSNR-Y {p:.2f} dB", fill=MUTED, font=f2)
            xx += im.shape[1] + pad
        name = f"crop_{iid}_{label}_{deg}.png"
        canvas.save(out / name)
        print("[figures]", name)


def _grid(tiles, path: Path, cols: int, tile_w: int) -> None:
    """Slide-friendly grid: tiles of equal width, caption bar under each (name + region PSNR)."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        f1, f2 = ImageFont.truetype("DejaVuSans-Bold.ttf", 30), ImageFont.truetype("DejaVuSans.ttf", 26)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    ims = []
    for name, im, p in tiles:
        s = tile_w / im.shape[1]
        r = interp.resize(im, max(1, round(im.shape[0] * s)), tile_w, "nearest" if s >= 1 else "bicubic")
        ims.append((name, to_uint8(r), p))
    th = max(i.shape[0] for _, i, _ in ims)
    cap, pad = 48, 16
    rows = int(np.ceil(len(ims) / cols))
    canvas = Image.new("RGB", (cols * (tile_w + pad) + pad, rows * (th + cap + pad) + pad), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for k, (name, im, p) in enumerate(ims):
        r_, c_ = divmod(k, cols)
        x0, y0 = pad + c_ * (tile_w + pad), pad + r_ * (th + cap + pad)
        canvas.paste(Image.fromarray(im), (x0, y0))
        col = FAMILY.get(family(name), INK) if name not in ("Input (LR, nearest)", "Ground truth (HR)") else INK
        d.rectangle([x0, y0 + th + 2, x0 + 14, y0 + th + 40], fill=col)
        label = name.replace("FSRNet-L1-mixed", "FSRNet, degradation-trained (ours)").replace("FSRNet-L1", "FSRNet (ours)")
        d.text((x0 + 24, y0 + th + 6), label, fill=INK, font=f1)
        if p is not None:
            d.text((x0 + tile_w - 250, y0 + th + 8), f"{p:.2f} dB", fill=MUTED, font=f2)
    canvas.save(path)


def _stack_rows(tiles, path: Path, tile_w: int) -> None:
    """Wide regions (text lines): one method per row, label column on the left."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        f1, f2 = ImageFont.truetype("DejaVuSans-Bold.ttf", 22), ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    rows = []
    for name, im, p in tiles:
        s = tile_w / im.shape[1]
        r = interp.resize(im, max(1, round(im.shape[0] * s)), tile_w, "nearest" if s >= 1 else "bicubic")
        rows.append((name, to_uint8(r), p))
    lab, pad = 300, 8
    H = sum(r.shape[0] for _, r, _ in rows) + pad * (len(rows) + 1)
    canvas = Image.new("RGB", (lab + tile_w + 2 * pad, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    yy = pad
    for name, im, p in rows:
        canvas.paste(Image.fromarray(im), (lab + pad, yy))
        col = FAMILY.get(family(name), INK) if name not in ("Input (LR, nearest)", "Ground truth (HR)") else INK
        d.rectangle([lab - 4, yy, lab, yy + im.shape[0]], fill=col)
        d.text((pad, yy + 4), name, fill=INK, font=f1)
        if p is not None:
            d.text((pad, yy + 32), f"region PSNR {p:.2f} dB", fill=MUTED, font=f2)
        yy += im.shape[0] + pad
    canvas.save(path)


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def charts(run: Path, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    m = pd.read_csv(run / "metrics_per_image.csv").drop_duplicates(["image_id", "method", "scale", "degradation"], keep="last")
    m["source"] = np.where(m.image_id.str.startswith("poster"), "Posters (16)", "DIV2K photos (50)")
    facts = {}
    order = ["Nearest", "Bilinear", "Bicubic", "SRCNN", "FSRNet-L1", "Real-ESRGAN"]

    # 1. PSNR by method, posters vs photos, x3 clean
    d = m[(m.scale == 3) & (m.degradation == "bicubic") & m.method.isin(order)]
    g = d.groupby(["source", "method"]).psnr_y.mean().unstack(0).loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=False)
    for ax, src in zip(axes, ["DIV2K photos (50)", "Posters (16)"]):
        v = g[src]
        bars = ax.bar(range(len(order)), v.values, color=[FAMILY[family(o)] for o in order], width=0.62,
                      edgecolor="white", linewidth=2)
        ax.set_ylim(v.min() - 2.5, v.max() + 1.2)
        for i, val in enumerate(v.values):
            ax.text(i, val + 0.15, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(len(order)), [o.replace("FSRNet-L1", "FSRNet\n(ours)").replace("Real-ESRGAN", "Real-ESRGAN\n(pretrained)") for o in order], fontsize=9)
        ax.set_title(src, fontsize=12, color=INK)
        ax.set_ylabel("PSNR-Y (dB) ↑", color=MUTED)
        _style(ax)
    fig.suptitle("3× (720p→4K), clean bicubic inputs: pixel fidelity", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out / "chart_psnr_by_source.png", dpi=200)
    plt.close(fig)

    # 2. Perception-distortion plane per source (x3 clean)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    meths = ["Bicubic", "SRCNN", "FSRNet-L1", "FSRNet-L1-mixed", "Real-ESRGAN"]
    for ax, src in zip(axes, ["DIV2K photos (50)", "Posters (16)"]):
        s = d[d.source == src] if False else m[(m.scale == 3) & (m.degradation == "bicubic") & (m.source == src)]
        s = s[s.method.isin(meths)].groupby("method")[["psnr_y", "lpips"]].mean()
        for meth, r in s.iterrows():
            ax.scatter(r.psnr_y, r.lpips, s=90, color=FAMILY[family(meth)], edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(meth.replace("FSRNet-L1-mixed", "FSRNet-mixed"), (r.psnr_y, r.lpips), xytext=(6, 5),
                        textcoords="offset points", fontsize=9, color=INK)
        ax.set_xlabel("PSNR-Y (dB) → more faithful", color=MUTED)
        ax.set_ylabel("LPIPS ↓ more perceptually similar", color=MUTED)
        ax.set_title(src, fontsize=12, color=INK)
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.margins(x=0.18, y=0.15)
    fig.suptitle("Perception–distortion plane (3×, clean inputs): best = bottom-right", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out / "chart_perception_distortion.png", dpi=200)
    plt.close(fig)

    # 3. Degradation robustness (A2): clean vs mixed inputs, x3, all 66
    meths = ["Bicubic", "FSRNet-L1", "FSRNet-L1-mixed", "Real-ESRGAN"]
    t = m[(m.scale == 3) & m.method.isin(meths)].groupby(["degradation", "method"]).psnr_y.mean().unstack(0).loc[meths]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = np.arange(len(meths))
    ax.bar(xs - 0.19, t["bicubic"], 0.36, color="#9ec3ef", label="clean (bicubic) inputs", edgecolor="white", linewidth=1.5)
    ax.bar(xs + 0.19, t["mixed"], 0.36, color="#1f5fa8", label="blur + noise + JPEG inputs", edgecolor="white", linewidth=1.5)
    for i, mm in enumerate(meths):
        ax.text(i - 0.19, t.loc[mm, "bicubic"] + 0.1, f"{t.loc[mm, 'bicubic']:.1f}", ha="center", fontsize=8.5)
        ax.text(i + 0.19, t.loc[mm, "mixed"] + 0.1, f"{t.loc[mm, 'mixed']:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs, ["Bicubic", "FSRNet\n(clean-trained)", "FSRNet\n(degradation-trained)", "Real-ESRGAN"], fontsize=9)
    ax.set_ylim(27, 34.6)
    ax.set_ylabel("PSNR-Y (dB) ↑", color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Ablation A2: training on degraded inputs (3×, all 66 test images)", fontsize=12, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "chart_ablation_degradation.png", dpi=200)
    plt.close(fig)

    # 4. 3x vs 6x (posters + photos), clean
    meths = ["Bicubic", "SRCNN", "FSRNet-L1", "FSRNet-L1 3x>3x>down", "Real-ESRGAN"]
    t = m[(m.degradation == "bicubic") & m.method.isin(meths)].groupby(["scale", "method"]).psnr_y.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, mm in enumerate(meths):
        if 3 in t.columns and not np.isnan(t.loc[mm].get(3, np.nan)):
            ax.plot([3, 6], [t.loc[mm, 3], t.loc[mm, 6]], marker="o", lw=2, ms=7, color=FAMILY[family(mm)],
                    alpha=1 if "3x>3x" not in mm else 0.6)
            ax.annotate(f"{mm} ({t.loc[mm, 6]:.2f})", (6, t.loc[mm, 6]), xytext=(8, 0), textcoords="offset points",
                        fontsize=8.5, va="center")
        else:
            ax.scatter([6], [t.loc[mm, 6]], marker="D", s=50, color=FAMILY["ours"], zorder=3)
            ax.annotate(f"FSRNet 3×→3×→↓ ({t.loc[mm, 6]:.2f})", (6, t.loc[mm, 6]), xytext=(8, 8),
                        textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xticks([3, 6], ["3×  (720p→4K)", "6×  (720p→8K)"])
    ax.set_xlim(2.6, 8.4)
    ax.set_ylabel("PSNR-Y (dB) ↑", color=MUTED)
    ax.set_title("3× vs 6×: all methods lose 3–5 dB (66 images, clean inputs)", fontsize=12, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "chart_3x_vs_6x.png", dpi=200)
    plt.close(fig)

    # 5. Region (ROI) PSNR on posters, x3 clean
    r = pd.read_csv(run / "roi_metrics.csv")
    r = r[(r.scale == 3) & (r.degradation == "bicubic") & r.image_id.str.startswith("poster")]
    labels = ["text_small", "text_logo", "face", "edges", "texture"]
    meths = ["Bicubic", "SRCNN", "FSRNet-L1", "Real-ESRGAN"]
    t = r[r.method.isin(meths) & r.roi.isin(labels)].groupby(["roi", "method"]).psnr_y.mean().unstack(1).loc[labels, meths]
    n = r[(r.method == "Bicubic") & r.roi.isin(labels)].groupby("roi").size()
    fig, ax = plt.subplots(figsize=(10, 3.9))
    xs = np.arange(len(labels))
    wdt = 0.2
    for j, mm in enumerate(meths):
        ax.bar(xs + (j - 1.5) * wdt, t[mm], wdt * 0.92, color=FAMILY[family(mm)] if mm != "SRCNN" else "#f5b48f",
               label=mm.replace("FSRNet-L1", "FSRNet (ours)"), edgecolor="white", linewidth=1)
    nice = {"text_small": "small text", "text_logo": "title / logo", "face": "faces", "edges": "edges / lines", "texture": "texture"}
    ax.set_xticks(xs, [f"{nice[l]}\n(n={n[l]})" for l in labels], fontsize=9.5)
    ax.set_ylim(20, 40)
    ax.set_ylabel("region PSNR-Y (dB) ↑", color=MUTED)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper left")
    ax.set_title("Detail regions on posters (3×, clean): our model gains most on text", fontsize=12, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "chart_roi_posters.png", dpi=200)
    plt.close(fig)
    facts["roi_posters_x3"] = t.round(2).to_dict()

    # 6. Runtime table chart (720p -> 4K / 8K on T4)
    rt = pd.read_csv(run / "runtime.csv")
    rt = rt[(rt.input == "1280x720") & rt.method.isin(["Bicubic", "SRCNN", "FSRNet-L1", "Real-ESRGAN"])]
    facts["runtime_720p"] = rt[["method", "scale", "median_s", "peak_gpu_mb", "device"]].round(3).to_dict("records")
    return facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="outputs/eval/main")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--slides-only", action="store_true", help="only the slide-format grids")
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    facts = charts(run, out)
    (out / "facts.json").write_text(json.dumps(facts, indent=1, default=str))
    SUB = [("Bicubic", "bicubic", {}), ("FSRNet-L1", "cnn", {"checkpoint": "models/fsrnet_x3_l1_bicubic.pt"}),
           ("Real-ESRGAN", "realesrgan", {})]
    slide_specs = [
        {"id": "poster_1401918", "roi": "text_small", "sub": (0, 399), "layout": "grid", "methods": SUB, "cols": 1, "no_input": True},
        {"id": "poster_1166038", "roi": "text_logo", "deg": "mixed", "layout": "grid", "sub": (0, 252),
         "methods": [("Bicubic", "bicubic", {}), ("FSRNet-L1-mixed", "cnn", {"checkpoint": "models/fsrnet_x3_l1_mixed.pt"}),
                     ("Real-ESRGAN", "realesrgan", {})]},
        {"id": "poster_1329765", "roi": "face", "layout": "grid", "methods": SUB, "cols": 3, "tile_w": 600},
    ]
    if a.slides_only:
        region_crops(run, out, slide_specs, Path(a.processed))
        return 0
    if not a.no_crops:
        specs = [
            {"id": "poster_1401918", "roi": "text_small", "sub": (0, 399)},
            {"id": "poster_813161", "roi": "text_logo", "tile_h": 200},
            {"id": "poster_1253488", "roi": "text_small", "tile_w": 900},
            {"id": "poster_1329765", "roi": "face", "tile_h": 300},
            {"id": "poster_1253488", "roi": "face", "tile_h": 300},
            {"id": "poster_1166038", "roi": "text_logo", "deg": "mixed", "tile_h": 180,
             "methods": [("Bicubic", "bicubic", {}),
                         ("FSRNet-L1", "cnn", {"checkpoint": "models/fsrnet_x3_l1_bicubic.pt"}),
                         ("FSRNet-L1-mixed", "cnn", {"checkpoint": "models/fsrnet_x3_l1_mixed.pt"}),
                         ("Real-ESRGAN", "realesrgan", {})]},
        ]
        region_crops(run, out, specs, Path(a.processed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
