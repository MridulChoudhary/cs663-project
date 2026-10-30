"""Create a tiny, freely usable sample dataset for smoke tests (no downloads).

Generates procedural "poster-like" images (gradients, typography, fine lines,
textures, shapes) plus a few public-domain / CC0 photos bundled with
scikit-image.  Output layout matches configs/sample_dataset.yaml:

    data/sample/train/*.png    data/sample/val/*.png
    data/sample/posters/*.png + tags.csv          data/sample/real_world/*.png

python scripts/make_sample_data.py [--out data/sample] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORDS = ["NIGHTFALL", "THE LAST ORBIT", "ECHOES", "PAPER MOON", "AFTERGLOW", "SILENT CITY",
         "NORTHBOUND", "GLASS HARBOR", "MIDNIGHT REEL", "STARFIELD"]


def _font(size: int, bold: bool = True):
    names = ["DejaVuSans-Bold.ttf", "DejaVuSerif-Bold.ttf", "arialbd.ttf", "arial.ttf"] if bold else \
            ["DejaVuSans.ttf", "arial.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def gradient(h, w, rng):
    c1, c2 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    t = np.linspace(0, 1, h)[:, None, None]
    ang = rng.uniform(0, 1)
    t = (t * (1 - ang) + np.linspace(0, 1, w)[None, :, None] * ang)
    return (c1 * (1 - t) + c2 * t).astype(np.uint8)


def texture(h, w, rng, octaves=5):
    out = np.zeros((h, w), np.float32)
    for o in range(octaves):
        s = 2 ** (o + 2)
        g = rng.random((s, s)).astype(np.float32)
        out += np.asarray(Image.fromarray(g).resize((w, h), Image.BICUBIC)) / (o + 1)
    out = (out - out.min()) / (np.ptp(out) + 1e-6)
    return out


def poster(h, w, rng, kind):
    img = Image.fromarray(gradient(h, w, rng))
    d = ImageDraw.Draw(img)
    cats = ["poster"]
    if kind in ("lines", "mixed"):
        for _ in range(rng.integers(20, 60)):
            x0, y0 = rng.integers(0, w), rng.integers(0, h)
            d.line([x0, y0, x0 + rng.integers(-w, w), y0 + rng.integers(-h, h)],
                   fill=tuple(int(v) for v in rng.integers(0, 256, 3)), width=int(rng.integers(1, 3)))
        cx, cy = w // 2, int(h * 0.4)
        for r in range(10, min(h, w) // 3, int(rng.integers(5, 9))):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=1)
        cats.append("fine_lines")
    if kind in ("texture", "mixed"):
        t = texture(h, w, rng)[..., None]
        arr = np.asarray(img).astype(np.float32)
        arr = arr * (0.6 + 0.4 * t) + rng.normal(0, 6, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        cats.append("texture")
    if kind in ("shapes", "mixed"):
        for _ in range(rng.integers(4, 9)):
            x0, y0 = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
            s = int(rng.integers(30, max(31, min(h, w) // 3)))
            col = tuple(int(v) for v in rng.integers(0, 256, 3))
            (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x0 + s, y0 + s], fill=col)
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        d = ImageDraw.Draw(img)
        cats.append("illustration")
    # typography: title, tagline and tiny credits block (classic poster structure)
    title = WORDS[int(rng.integers(len(WORDS)))]
    f_big = _font(max(18, w // 9))
    tw = d.textlength(title, font=f_big)
    d.text(((w - tw) / 2, h * 0.72), title, font=f_big, fill=(255, 255, 255), stroke_width=2,
           stroke_fill=(0, 0, 0))
    f_small = _font(max(9, w // 55), bold=False)
    for i in range(4):
        line = "DIRECTED BY A. NAME  PRODUCED BY B. NAME  MUSIC BY C. NAME  2026"[: 30 + 8 * i]
        d.text((w * 0.08, h * 0.86 + i * (w // 45)), line, font=f_small, fill=(230, 230, 230))
    cats.append("text_logo")
    return np.asarray(img.convert("RGB")), cats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    rng = np.random.default_rng(a.seed)
    kinds = ["lines", "texture", "shapes", "mixed"]
    sizes = [(576, 384), (540, 360), (432, 648), (480, 480)]   # (H, W), all divisible by 6

    for split, n in (("train", 12), ("val", 3)):
        (out / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            h, w = sizes[i % len(sizes)]
            arr, _ = poster(h, w, rng, kinds[i % len(kinds)])
            Image.fromarray(arr).save(out / split / f"synth_{split}_{i:02d}.png")

    (out / "posters").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, kind in enumerate(kinds + ["mixed", "lines"]):
        h, w = sizes[i % len(sizes)]
        arr, cats = poster(h, w, rng, kind)
        name = f"synth_poster_{i:02d}.png"
        Image.fromarray(arr).save(out / "posters" / name)
        rows.append({"filename": name, "categories": ";".join(cats), "license": "procedurally generated (project)",
                     "notes": f"synthetic {kind}"})
    try:  # public-domain / CC0 photos bundled with scikit-image
        import skimage.data as skd
        for name, cats, lic in (("astronaut", "photographic", "public domain (NASA), via scikit-image"),
                                ("coffee", "photographic;texture", "CC0, via scikit-image"),
                                ("chelsea", "photographic;texture", "CC0, via scikit-image")):
            Image.fromarray(getattr(skd, name)()).save(out / "posters" / f"{name}.png")
            rows.append({"filename": f"{name}.png", "categories": cats, "license": lic, "notes": "photo"})
        (out / "real_world").mkdir(parents=True, exist_ok=True)
        rocket = Image.fromarray(skd.rocket())
        rocket.resize((rocket.width // 3, rocket.height // 3), Image.BICUBIC).save(
            out / "real_world" / "rocket_small.jpg", quality=70)
    except Exception as e:  # scikit-image optional
        print("scikit-image photos skipped:", e)
    with open(out / "posters" / "tags.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "categories", "license", "notes"])
        w.writeheader()
        w.writerows(rows)
    print(f"sample data written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
