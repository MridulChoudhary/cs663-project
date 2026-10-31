"""Dataset preparation: manifest -> splits -> train sub-images -> fixed eval pairs.

Everything is written under ``processed_dir`` and is resumable (existing files
are kept).  The output layout is::

    processed/
      manifest.csv                 one row per source image (+ split)
      splits/{train,val,test,real_world}.txt
      dataset_summary.json         counts by split / source / category
      prepare_log.txt
      contact_sheet.png
      train_sub/<id>_<y>_<x>.png   HR training sub-images (train split only)
      <split>/HR/<id>.png          HR ground truth, mod-cropped to lcm(scales)
      <split>/x<s>/<deg>/<id>.png  LR inputs, one folder per scale and degradation
      <split>/x<s>/<deg>/params.jsonl   exact degradation parameters per image
"""
from __future__ import annotations

import json
from collections import Counter
from math import lcm
from pathlib import Path

import numpy as np

from faithful_sr.data import manifest as mf
from faithful_sr.data.degradation import degrade, get_preset, image_rng
from faithful_sr.data.patches import extract_subimages
from faithful_sr.models.interpolation import downscale, modcrop, resize, upscale
from faithful_sr.utils.config import environment_info, write_json
from faithful_sr.utils.image_io import ImageLoadError, flatten_alpha, load_image, save_image
from faithful_sr.utils.visual import labelled_grid, save


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


def preprocess_tag(scfg: dict) -> str:
    parts = []
    if scfg.get("max_long_side"):
        parts.append(f"long_side<={scfg['max_long_side']}")
    if scfg.get("alpha_background") is not None:
        parts.append(f"alpha->{scfg['alpha_background']}")
    if scfg.get("lr_downscale"):
        parts.append(f"real_world_lr=/{scfg['lr_downscale']}(<= {scfg.get('lr_max_long_side', 'inf')})")
    if scfg.get("reference_upscaled"):
        parts.append("file_is_third_party_upscale(reference_only)")
    return ";".join(parts)


def load_source_image(row: mf.ManifestRow, root: Path, scfg: dict) -> np.ndarray:
    """Load a manifest image with its source's recorded preprocessing (cap size, flatten alpha)."""
    li = load_image(root / row.rel_path, max_pixels=int(scfg.get("max_pixels", 250_000_000)),
                    max_long_side=scfg.get("max_long_side"))
    bg = scfg.get("alpha_background", [0.0, 0.0, 0.0])
    return flatten_alpha(li, bg)


def detail_psnr(img: np.ndarray, factor: int = 2, crop: int = 1024) -> float:
    """Quality screen: PSNR(Y) between an image and its x``factor`` down-up version (central crop).

    Very high values (> ~45 dB) mean almost no energy above 1/``factor`` of Nyquist: the image is
    either very flat graphic art or was itself upscaled - a weak ground truth at this size.
    """
    from faithful_sr.evaluation.metrics import psnr
    h, w = img.shape[:2]
    c = min(crop, h, w) // factor * factor
    y, x = (h - c) // 2, (w - c) // 2
    a = img[y:y + c, x:x + c]
    b = upscale(downscale(a, factor, "bicubic"), factor, "bicubic")
    return psnr(b, a)


def build_manifest(cfg: dict, root: Path, log) -> list[mf.ManifestRow]:
    rows: list[mf.ManifestRow] = []
    for s in cfg["sources"]:
        rows += mf.scan_source(
            s["name"], root / s["dir"], root, s.get("split", "auto"), s.get("license", ""),
            s.get("categories", []), s.get("has_gt", True),
            root / s["tags_file"] if s.get("tags_file") else None, cfg.get("min_size", 96), log)
        for r in rows:
            if r.source == s["name"]:
                r.preprocess = preprocess_tag(s)
        if s.get("split", "auto") == "auto":
            src_rows = [r for r in rows if r.source == s["name"] and r.split == "auto"]
            mf.assign_auto_splits(src_rows, s.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1}),
                                  cfg.get("seed", 663))
    n = mf.mark_duplicates(rows, cfg.get("near_duplicate_hamming", 3), log)
    log(f"[manifest] duplicates excluded: {n}")
    mf.check_leakage(rows)
    return rows


def prepare(cfg: dict, root: Path) -> dict:
    out = root / cfg.get("processed_dir", "data/processed")
    log = Logger(out / "prepare_log.txt")
    log(f"=== prepare_dataset {environment_info()['timestamp_utc']} ===")
    rows = build_manifest(cfg, root, log)
    mf.write_manifest(rows, out / "manifest.csv")
    for split in mf.SPLITS:
        ids = [r.image_id for r in rows if r.split == split]
        (out / "splits").mkdir(parents=True, exist_ok=True)
        (out / "splits" / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    counts = Counter(r.split for r in rows)
    log(f"[splits] {dict(counts)}")
    for need in cfg.get("require_splits", []):
        if counts.get(need, 0) == 0:
            raise RuntimeError(f"Split '{need}' is empty - check the source directories in the config")

    scales = list(cfg.get("scales", [3, 6]))
    crop_mult = lcm(*scales)
    seed = int(cfg.get("seed", 663))
    scfgs = {s["name"]: s for s in cfg["sources"]}
    load_hr = lambda r: load_source_image(r, root, scfgs.get(r.source, {}))  # noqa: E731

    # ---- quality screen (sources with quality_screen: true)
    qrows = []
    qpath = out / "quality_screen.csv"
    done_q = set(pd_read_ids(qpath))
    for r in rows:
        if not scfgs.get(r.source, {}).get("quality_screen") or r.split == "excluded" or r.image_id in done_q:
            continue
        img = load_hr(r)
        qrows.append({"image_id": r.image_id, "split": r.split, "width": img.shape[1], "height": img.shape[0],
                      "detail_psnr_2x": round(detail_psnr(img, 2), 2), "detail_psnr_4x": round(detail_psnr(img, 4), 2)})
    if qrows:
        import csv
        new = not qpath.exists()
        with open(qpath, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(qrows[0]))
            if new:
                w.writeheader()
            w.writerows(qrows)
        flagged = [q["image_id"] for q in qrows if q["detail_psnr_2x"] > cfg.get("quality_flag_db", 45)]
        log(f"[quality] screened {len(qrows)} images; little fine detail (>{cfg.get('quality_flag_db', 45)} dB): {flagged}")

    # ---- training sub-images (HR only; LR is generated on the fly)
    sub = cfg.get("subimage", {"size": 480, "step": 480})
    sub_dir = out / "train_sub"
    sub_dir.mkdir(parents=True, exist_ok=True)
    n_sub = 0
    for r in mf.rows_for_split(rows, "train"):
        if any(sub_dir.glob(f"{r.image_id}_*.png")):
            n_sub += len(list(sub_dir.glob(f"{r.image_id}_*.png")))
            continue
        try:
            hr = load_hr(r)
        except ImageLoadError as e:
            log(f"[train_sub] skip {r.image_id}: {e}")
            continue
        for y, x, tile in extract_subimages(hr, sub["size"], sub["step"]):
            save_image(tile, sub_dir / f"{r.image_id}_{y:05d}_{x:05d}.png")
            n_sub += 1
    log(f"[train_sub] {n_sub} sub-images of {sub['size']}px (step {sub['step']}) in {sub_dir}")

    # ---- fixed evaluation pairs
    eval_degs = cfg.get("eval_degradations", ["bicubic"])
    for split in ("val", "test"):
        split_rows = [r for r in mf.rows_for_split(rows, split) if r.has_gt]
        for r in split_rows:
            hr_path = out / split / "HR" / f"{r.image_id}.png"
            hr = None
            for s in scales:
                for dname in eval_degs:
                    lr_path = out / split / f"x{s}" / dname / f"{r.image_id}.png"
                    if lr_path.exists() and hr_path.exists():
                        continue
                    if hr is None:
                        hr = modcrop(load_hr(r), crop_mult)
                        save_image(hr, hr_path)
                    variant = scales.index(s) * 100 + eval_degs.index(dname)
                    lr, params = degrade(hr, s, get_preset(cfg.get("degradations", {}).get(dname, dname)),
                                         image_rng(seed, r.image_id, variant))
                    save_image(lr, lr_path)
                    params["image_id"] = r.image_id
                    with open(lr_path.parent / "params.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(params) + "\n")
        log(f"[{split}] {len(split_rows)} images x scales {scales} x degradations {eval_degs}")

    # ---- real-world inputs (no ground truth) + optional third-party reference upscales
    n_rw = 0
    for r in mf.rows_for_split(rows, "real_world"):
        scfg = scfgs.get(r.source, {})
        lr_path = out / "real_world" / "LR" / f"{r.image_id}.png"
        if lr_path.exists():
            n_rw += 1
            continue
        try:
            li = load_image(root / r.rel_path, max_pixels=int(scfg.get("max_pixels", 250_000_000)))
        except ImageLoadError as e:
            log(f"[real_world] skip {r.image_id}: {e}")
            continue
        raw = flatten_alpha(li, scfg.get("alpha_background", [0.0, 0.0, 0.0]))
        h, w = raw.shape[:2]
        f = float(scfg.get("lr_downscale", 1))
        cap = scfg.get("lr_max_long_side")
        sc = 1.0 / f
        if cap and max(h, w) * sc > cap:
            sc = cap / max(h, w)
        lh, lw = max(1, round(h * sc)), max(1, round(w * sc))
        lr = resize(raw, lh, lw, "bicubic") if (lh, lw) != (h, w) else raw
        save_image(lr, lr_path)
        if scfg.get("reference_upscaled"):
            s_rw = int(cfg.get("real_world_scale", 3))
            save_image(resize(raw, lh * s_rw, lw * s_rw, "bicubic"), out / "real_world" / "REF" / f"{r.image_id}.png")
        n_rw += 1
    if n_rw:
        log(f"[real_world] {n_rw} LR inputs in {out / 'real_world' / 'LR'}")

    for dname in eval_degs:
        write_json(get_preset(cfg.get("degradations", {}).get(dname, dname)).to_dict(),
                   out / "degradation_configs" / f"{dname}.json")

    summary = {
        "environment": environment_info(), "seed": seed, "scales": scales, "hr_modcrop": crop_mult,
        "split_counts": dict(counts),
        "by_source_split": {f"{k[0]}/{k[1]}": v for k, v in Counter((r.source, r.split) for r in rows).items()},
        "category_counts_by_split": {
            sp: dict(Counter(c for r in rows if r.split == sp for c in r.category_list)) for sp in mf.SPLITS},
        "train_subimages": n_sub, "subimage": sub, "eval_degradations": eval_degs,
    }
    write_json(summary, out / "dataset_summary.json")
    make_contact_sheet(out, rows, scales[0], eval_degs, log)
    log("=== done ===")
    return summary


def make_contact_sheet(out: Path, rows, scale: int, degs, log, n: int = 4) -> None:
    """HR | LR(each degradation, shown nearest-upscaled) for a few test/val images."""
    picks = [r for r in rows if r.split == "test" and r.has_gt][:n] or \
            [r for r in rows if r.split == "val" and r.has_gt][:n]
    if not picks:
        return
    imgs, labels = [], []
    for r in picks:
        split = r.split
        hr = load_image(out / split / "HR" / f"{r.image_id}.png").rgb
        imgs.append(hr)
        labels.append(f"{r.image_id[:22]} HR")
        for d in degs:
            lr = load_image(out / split / f"x{scale}" / d / f"{r.image_id}.png").rgb
            # show the central 1/4 so degradation is visible at contact-sheet size
            h, w = lr.shape[:2]
            imgs.append(upscale(lr[h // 4: 3 * h // 4, w // 4: 3 * w // 4], scale, "nearest"))
            labels.append(f"LR x{scale} {d} (centre)")
    sheet = labelled_grid(imgs, labels, cols=1 + len(degs), cell=220,
                          title=f"Dataset contact sheet: HR and x{scale} LR inputs per degradation")
    save(sheet, out / "contact_sheet.png")
    log(f"[contact_sheet] {out / 'contact_sheet.png'}")


def pd_read_ids(path: Path) -> list[str]:
    if not Path(path).exists():
        return []
    import csv
    with open(path, newline="") as f:
        return [r["image_id"] for r in csv.DictReader(f)]


def load_split_ids(processed: Path, split: str) -> list[str]:
    p = Path(processed) / "splits" / f"{split}.txt"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def category_lookup(processed: Path) -> dict[str, list[str]]:
    return {r.image_id: r.category_list for r in mf.read_manifest(Path(processed) / "manifest.csv")}


__all__ = ["prepare", "build_manifest", "load_split_ids", "category_lookup", "np"]
