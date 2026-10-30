"""Benchmark runner: every method x scale x degradation x test image.

Outputs (all under ``output_dir``)::

    config_used.yaml / environment.json
    status.csv                 one row per method/scale/degradation: ran | not run (+ reason)
    metrics_per_image.csv      every score, with image categories
    roi_metrics.csv            ROI-level scores (detail-preservation analysis)
    rois_used.json             exact ROI boxes (repeatability)
    visuals/<scale>/<deg>/<id>_grid.png      labelled full-image comparisons
    visuals/<scale>/<deg>/<id>_roi_<label>.png  zoomed ROI crops + error maps
    sr/<scale>/<deg>/<method>/<id>.png       SR outputs for the visual subset

A method that cannot run (missing checkpoint/weights/dependency) is recorded as
"not run" with the exact reason - never silently skipped, never filled in.
"""
from __future__ import annotations

import csv
import json
import re
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from faithful_sr.data.manifest import read_manifest
from faithful_sr.evaluation import metrics as M
from faithful_sr.evaluation.region_analysis import load_manual_rois, roi_metrics, rois_for_image
from faithful_sr.inference.upscale import get_method
from faithful_sr.models import interpolation as interp
from faithful_sr.utils.config import environment_info, write_json
from faithful_sr.utils.image_io import load_image, save_image
from faithful_sr.utils.visual import crop, draw_boxes, error_map, labelled_grid, save


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


class CsvWriter:
    def __init__(self, path: Path):
        self.path, self.f, self.w = path, None, None

    def write(self, row: dict) -> None:
        if self.w is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.path, "w", newline="", encoding="utf-8")
            self.w = csv.DictWriter(self.f, fieldnames=list(row), extrasaction="ignore")
            self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def method_entries(cfg: dict, scale: int) -> list[dict]:
    entries = list(cfg.get("methods", []))
    if scale == 6:
        entries += cfg.get("six_x_methods", [])
    return [e for e in entries if scale in e.get("scales", [scale])]


def run_benchmark(cfg: dict, log=print) -> dict:
    processed = Path(cfg.get("processed_dir", "data/processed"))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(environment_info(), out / "environment.json")
    split = cfg.get("split", "test")
    rows = {r.image_id: r for r in read_manifest(processed / "manifest.csv")}
    ids = sorted(p.stem for p in (processed / split / "HR").glob("*.png"))
    if cfg.get("max_images"):
        ids = ids[: int(cfg["max_images"])]
    if not ids:
        raise RuntimeError(f"No {split} images in {processed / split / 'HR'} - run prepare_dataset first")
    visual_ids = set(cfg.get("visual_ids") or ids[: int(cfg.get("n_visual", 6))])
    manual_rois = load_manual_rois(cfg.get("rois"))
    lp = M.try_lpips() if "lpips" in cfg.get("metrics", []) else None
    if "lpips" in cfg.get("metrics", []) and lp is None:
        log("[benchmark] LPIPS unavailable (pip install lpips; needs AlexNet weights) - column left empty")

    per_img, per_roi, status = CsvWriter(out / "metrics_per_image.csv"), CsvWriter(out / "roi_metrics.csv"), []
    rois_used: dict[str, list] = {}
    hr_cache: dict[str, np.ndarray] = {}

    def get_hr(iid):
        if iid not in hr_cache:
            hr_cache.clear()
            hr_cache[iid] = load_image(processed / split / "HR" / f"{iid}.png").rgb
        return hr_cache[iid]

    for scale in cfg.get("scales", [3]):
        for deg in cfg.get("degradations", ["bicubic"]):
            lr_dir = processed / split / f"x{scale}" / deg
            if not lr_dir.exists():
                status.append({"method": "*", "scale": scale, "degradation": deg, "status": "not run",
                               "reason": f"missing {lr_dir}"})
                continue
            fns = {}
            for e in method_entries(cfg, scale):
                label = e["label"]
                try:
                    fns[label] = get_method(e["method"], **e.get("kwargs", {}))
                    status.append({"method": label, "scale": scale, "degradation": deg, "status": "ran", "reason": ""})
                except Exception as ex:  # missing checkpoint / weights / dependency
                    status.append({"method": label, "scale": scale, "degradation": deg, "status": "not run",
                                   "reason": f"{type(ex).__name__}: {ex}"})
                    log(f"[benchmark] {label} x{scale} {deg}: NOT RUN - {ex}")
            for iid in ids:
                hr = get_hr(iid)
                lr = load_image(lr_dir / f"{iid}.png").rgb
                cats = rows[iid].category_list if iid in rows else []
                rois = rois_used.setdefault(iid, rois_for_image(iid, hr, manual_rois))
                outputs: dict[str, np.ndarray] = {}
                for label, fn in fns.items():
                    t0 = time.perf_counter()
                    try:
                        sr = fn(lr, scale)
                    except Exception as ex:
                        log(f"[benchmark] {label} failed on {iid}: {ex}")
                        status.append({"method": label, "scale": scale, "degradation": deg, "status": "error",
                                       "reason": f"{iid}: {ex}\n{traceback.format_exc(limit=1)}"})
                        continue
                    rt = time.perf_counter() - t0
                    if sr.shape != hr.shape:
                        raise RuntimeError(f"{label} returned {sr.shape}, expected {hr.shape}")
                    rec = {"image_id": iid, "categories": ";".join(cats), "method": label, "scale": scale,
                           "degradation": deg, "psnr_y": M.psnr(sr, hr, scale), "ssim_y": M.ssim(sr, hr, scale),
                           "psnr_rgb": M.psnr(sr, hr, scale, y_only=False),
                           "edge_psnr": M.edge_psnr(sr, hr, scale), "overshoot": M.overshoot(sr, hr, scale),
                           "lpips": lp(sr, hr, scale) if lp else float("nan"), "runtime_s": rt,
                           "width": hr.shape[1], "height": hr.shape[0]}
                    per_img.write(rec)
                    for r in rois:
                        per_roi.write({"image_id": iid, "categories": ";".join(cats), "method": label,
                                       "scale": scale, "degradation": deg, "roi": r["label"],
                                       "roi_source": r["source"], **roi_metrics(sr, hr, r["box"])})
                    if iid in visual_ids:
                        outputs[label] = sr
                        if cfg.get("save_outputs", True):
                            save_image(sr, out / "sr" / f"x{scale}" / deg / slug(label) / f"{iid}.png")
                if iid in visual_ids and outputs:
                    save_visuals(out / "visuals" / f"x{scale}" / deg, iid, lr, hr, outputs, rois, scale)
                log(f"[benchmark] x{scale} {deg} {iid}: {len(outputs) or len(fns)} methods")
    per_img.close()
    per_roi.close()
    with open(out / "status.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scale", "degradation", "status", "reason"])
        w.writeheader()
        w.writerows(status)
    write_json(rois_used, out / "rois_used.json")
    return {"output_dir": str(out), "n_images": len(ids), "status": status}


def save_visuals(vdir: Path, iid: str, lr, hr, outputs: dict, rois: list, scale: int) -> None:
    lr_up = interp.upscale(lr, scale, "nearest")
    imgs = [lr_up] + list(outputs.values()) + [hr]
    labels = [f"LR input (x{scale}, nearest)"] + [f"{k}  {M.psnr(v, hr, scale):.2f} dB" for k, v in outputs.items()] + ["HR ground truth"]
    boxed_hr = draw_boxes(hr, [r["box"] for r in rois], [r["label"] for r in rois], width=max(2, hr.shape[1] // 300))
    imgs[-1] = boxed_hr
    save(labelled_grid(imgs, labels, cols=min(4, len(imgs)), cell=320, title=f"{iid}  x{scale}"), vdir / f"{iid}_grid.png")
    for r in rois:
        crops = [crop(lr_up, r["box"])] + [crop(v, r["box"]) for v in outputs.values()] + [crop(hr, r["box"])]
        errs = [error_map(crop(v, r["box"]), crop(hr, r["box"])) for v in outputs.values()]
        names = ["LR (nearest)"] + list(outputs) + ["HR"]
        grid = labelled_grid(crops + errs, names + [f"|err| {k}" for k in outputs],
                             cols=len(crops), cell=200, title=f"{iid} ROI '{r['label']}' ({r['source']})  x{scale}")
        save(grid, vdir / f"{iid}_roi_{r['label']}.png")


# ---------------------------------------------------------------- runtime
def run_runtime(cfg: dict, log=print) -> list[dict]:
    """E8: runtime/memory on synthetic inputs of fixed sizes (excludes model loading)."""
    import torch
    out = Path(cfg["output_dir"])
    sizes = cfg.get("runtime_sizes", [[320, 180], [1280, 720]])
    reps, warm = int(cfg.get("runtime_repeats", 3)), int(cfg.get("runtime_warmup", 1))
    rng = np.random.default_rng(0)
    results = []
    for scale in cfg.get("scales", [3]):
        for e in method_entries(cfg, scale):
            try:
                fn = get_method(e["method"], **e.get("kwargs", {}))
            except Exception as ex:
                results.append({"method": e["label"], "scale": scale, "status": f"not run: {ex}"})
                continue
            for w, h in sizes:
                if e["method"] == "realesrgan" and w * h > 400_000 and not torch.cuda.is_available():
                    results.append({"method": e["label"], "scale": scale, "input": f"{w}x{h}",
                                    "status": "not run: skipped on CPU (too slow)"})
                    continue
                img = rng.random((h, w, 3)).astype(np.float32)
                cuda = torch.cuda.is_available() and e["method"] in ("cnn", "realesrgan")
                try:
                    for _ in range(warm):
                        fn(img, scale)
                    if cuda:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    times = []
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        fn(img, scale)
                        if cuda:
                            torch.cuda.synchronize()
                        times.append(time.perf_counter() - t0)
                    rec = {"method": e["label"], "scale": scale, "input": f"{w}x{h}", "output": f"{w*scale}x{h*scale}",
                           "median_s": float(np.median(times)), "min_s": float(np.min(times)),
                           "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                           "tile": e.get("kwargs", {}).get("tile", 0), "device": "cuda" if cuda else "cpu",
                           "repeats": reps, "warmup": warm, "status": "ran"}
                except Exception as ex:
                    rec = {"method": e["label"], "scale": scale, "input": f"{w}x{h}", "status": f"error: {ex}"}
                results.append(rec)
                log(f"[runtime] {rec}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runtime.csv", "w", newline="") as f:
        keys = sorted({k for r in results for k in r}, key=lambda k: ["method", "scale", "input", "output",
                      "median_s", "min_s", "peak_gpu_mb", "tile", "device", "repeats", "warmup", "status"].index(k)
                      if k in ["method", "scale", "input", "output", "median_s", "min_s", "peak_gpu_mb", "tile",
                               "device", "repeats", "warmup", "status"] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    write_json({"environment": environment_info(), "note": "model loading excluded; median of repeats after warm-up"},
               out / "runtime_info.json")
    return results


# ------------------------------------------------------------ real-world
def run_real_world(cfg: dict, log=print) -> None:
    """Qualitative only: no ground truth exists, so no scores are computed."""
    processed = Path(cfg.get("processed_dir", "data/processed"))
    root = Path(cfg.get("root", "."))
    rows = [r for r in read_manifest(processed / "manifest.csv") if r.split == "real_world"]
    out = Path(cfg["output_dir"]) / "real_world"
    if not rows:
        log("[real_world] no real-world images in manifest")
        return
    scale = int(cfg.get("real_world_scale", 3))
    fns = {}
    for e in method_entries(cfg, scale):
        try:
            fns[e["label"]] = get_method(e["method"], **e.get("kwargs", {}))
        except Exception as ex:
            log(f"[real_world] {e['label']}: not run - {ex}")
    notes = ["# Real-world images (qualitative only - no ground truth, no scores)\n",
             "Fill in observations per image: lettering changes, face changes, halos/ringing, "
             "invented-looking textures ('possible hallucination'), compression artefacts amplified.\n"]
    for r in rows:
        img = load_image(root / r.rel_path).rgb
        if max(img.shape[:2]) > 800:
            log(f"[real_world] {r.image_id} is large; using its centre 800px crop")
            h, w = img.shape[:2]
            y, x = max(0, (h - 800) // 2), max(0, (w - 800) // 2)
            img = img[y:y + 800, x:x + 800]
        outs = {k: f(img, scale) for k, f in fns.items()}
        h, w = img.shape[:2]
        cb = [w * scale // 3, h * scale // 3, w * scale // 4, h * scale // 4]
        imgs = [interp.upscale(img, scale, "nearest")] + list(outs.values())
        labels = ["input (nearest)"] + list(outs)
        save(labelled_grid([crop(i, cb) for i in imgs], labels, cols=len(imgs), cell=300,
                           title=f"{r.image_id} (real-world, x{scale}, centre crop) - qualitative"),
             out / f"{r.image_id}_crops.png")
        save(labelled_grid(imgs, labels, cols=min(4, len(imgs)), cell=320, title=f"{r.image_id} x{scale}"),
             out / f"{r.image_id}_grid.png")
        notes.append(f"\n## {r.image_id}\nSource/licence: {r.license}\n\n- Observations: _pending human review_\n")
    (out / "observations_template.md").write_text("".join(notes), encoding="utf-8")
    log(f"[real_world] {len(rows)} images -> {out}")
