"""Tiled inference for images too large to process in one pass.

Strategy: split the LR image into a grid of ``tile x tile`` *core* regions.
Each core is processed together with a halo of ``overlap`` LR pixels on every
side (clipped at the true image border), and only the core's part of the
output is kept.

Why this is seam-free: an output pixel of a convolutional network depends only
on LR pixels within half its receptive field.  If ``overlap >= RF/2`` every
kept output pixel sees exactly the same input as in whole-image inference, so
the tiled result equals the untiled one (up to float rounding).  At the true
image border the halo is clipped, so the network sees the same border as in
whole-image inference too.  For networks with very large receptive fields
(e.g. Real-ESRGAN's RRDB) a practical overlap is used and small seams remain
possible; ``tests/test_tiling.py`` measures this on flat, gradient and
pattern images.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def tile_grid(size: int, tile: int) -> list[tuple[int, int]]:
    starts = list(range(0, size, tile))
    return [(s, min(s + tile, size)) for s in starts]


def tiled_upscale(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], scale: int, tile: int,
                  overlap: int, progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Apply ``fn`` (LR float image -> scale-x float image) tile by tile."""
    if tile <= 0:
        return fn(img)
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    h, w = img.shape[:2]
    if h <= tile and w <= tile:
        return fn(img)
    out = np.zeros((h * scale, w * scale, img.shape[2]), np.float32)
    ys, xs = tile_grid(h, tile), tile_grid(w, tile)
    total, k = len(ys) * len(xs), 0
    for y0, y1 in ys:
        for x0, x1 in xs:
            hy0, hy1 = max(0, y0 - overlap), min(h, y1 + overlap)
            hx0, hx1 = max(0, x0 - overlap), min(w, x1 + overlap)
            res = fn(img[hy0:hy1, hx0:hx1])
            if res.shape[:2] != ((hy1 - hy0) * scale, (hx1 - hx0) * scale):
                raise RuntimeError("Tile function returned an unexpected size")
            oy, ox = (y0 - hy0) * scale, (x0 - hx0) * scale
            out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                res[oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
            k += 1
            if progress:
                progress(k, total)
    return out
