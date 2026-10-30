"""Labelled image grids, crops and error maps for reports and contact sheets."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from faithful_sr.models.interpolation import resize
from faithful_sr.utils.image_io import to_uint8


def _font(size: int = 14):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(img: np.ndarray, size: int, method: str = "nearest") -> np.ndarray:
    """Resize so the longer side equals ``size``.

    Enlargement uses ``method`` (nearest by default: shows the true pixels);
    reduction always uses anti-aliased bicubic so thumbnails do not alias.
    """
    h, w = img.shape[:2]
    s = size / max(h, w)
    return resize(img, max(1, round(h * s)), max(1, round(w * s)), method if s >= 1 else "bicubic")


def labelled_grid(images: Sequence[np.ndarray], labels: Sequence[str], cols: int,
                  cell: int = 256, title: Optional[str] = None, pad: int = 4,
                  method: str = "nearest") -> Image.Image:
    """Arrange float images in a grid with a caption under each."""
    font, label_h = _font(13), 20
    title_h = 26 if title else 0
    rows = int(np.ceil(len(images) / cols))
    W = cols * (cell + pad) + pad
    H = title_h + rows * (cell + label_h + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((pad, 4), title, fill=(0, 0, 0), font=_font(16))
    for i, (img, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        x0, y0 = pad + c * (cell + pad), title_h + pad + r * (cell + label_h + pad)
        tile = to_uint8(fit(img, cell, method)) if img.ndim == 3 else to_uint8(fit(img[..., None].repeat(3, 2), cell, method))
        canvas.paste(Image.fromarray(tile), (x0 + (cell - tile.shape[1]) // 2, y0 + (cell - tile.shape[0]) // 2))
        draw.text((x0, y0 + cell + 2), lab[:40], fill=(0, 0, 0), font=font)
    return canvas


def crop(img: np.ndarray, box: Sequence[int]) -> np.ndarray:
    """``box = (x, y, w, h)`` in pixel coordinates of ``img``."""
    x, y, w, h = map(int, box)
    return img[y:y + h, x:x + w]


def error_map(pred: np.ndarray, gt: np.ndarray, vmax: float = 0.25) -> np.ndarray:
    """Per-pixel mean absolute error rendered with a perceptually ordered colormap."""
    import matplotlib
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(axis=2)
    return matplotlib.colormaps["inferno"](np.clip(err / vmax, 0, 1))[..., :3].astype(np.float32)


def draw_boxes(img: np.ndarray, boxes: Sequence[Sequence[int]], labels: Sequence[str] = (),
               width: int = 3) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img))
    d = ImageDraw.Draw(pil)
    colours = [(230, 57, 70), (29, 120, 220), (40, 170, 90), (240, 160, 0)]
    for i, b in enumerate(boxes):
        x, y, w, h = map(int, b)
        col = colours[i % len(colours)]
        d.rectangle([x, y, x + w, y + h], outline=col, width=width)
        if i < len(labels):
            d.text((x + 4, y + 4), labels[i], fill=col, font=_font(max(12, width * 5)))
    return np.asarray(pil).astype(np.float32) / 255.0


def save(pil: Image.Image, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pil.save(p)
    return p
