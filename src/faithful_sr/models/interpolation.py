r"""Classical interpolation implemented from scratch in NumPy/SciPy.

Theory (see docs/methodology.md for the full derivation)
--------------------------------------------------------
Resampling a discrete image ``f[j]`` at continuous position ``x`` is a
convolution with a reconstruction kernel ``k``:

.. math::  \hat f(x) = \sum_j f[j]\, k(x - j)

The kernels implemented are

* nearest   : box,      ``k(t) = 1`` for ``|t| < 1/2``
* bilinear  : triangle, ``k(t) = 1 - |t|`` for ``|t| < 1``
* bicubic   : Keys (1981) cubic convolution with ``a = -0.5``
              (support 2; the value used by MATLAB ``imresize`` and Pillow)
* lanczos3  : ``sinc(t) sinc(t/3)`` for ``|t| < 3`` (only used for resizing a
              pretrained model's 4x output to the 3x/6x target size)

Coordinates use the *pixel-centre* convention: output pixel ``i`` maps to input
position ``c = (i + 0.5) / s - 0.5`` for scale factor ``s = out/in``.

When downsampling (``s < 1``) the kernel is stretched by ``1/s`` (and divided by
``1/s``) so that it acts as an anti-aliasing low-pass filter — exactly what
MATLAB ``imresize`` and Pillow do, and the standard way SR benchmarks create LR
images ("bicubic degradation").

Borders: kernel taps falling outside the image are dropped and the remaining
weights renormalised to sum to 1 (Pillow's convention).

Because every kernel is separable, the 2-D resize is two sparse matrix
products: ``out = W_h  @  img  @  W_w^T``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
import scipy.sparse as sp

METHODS = ("nearest", "bilinear", "bicubic", "lanczos3")


# ------------------------------------------------------------------ kernels
def box_kernel(t: np.ndarray) -> np.ndarray:
    t = np.abs(t)
    return np.where(t < 0.5, 1.0, np.where(t == 0.5, 0.5, 0.0))


def triangle_kernel(t: np.ndarray) -> np.ndarray:
    t = np.abs(t)
    return np.maximum(0.0, 1.0 - t)


def keys_cubic_kernel(t: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Keys cubic convolution kernel.

    k(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1            for |t| <= 1
         = a|t|^3 - 5a|t|^2 + 8a|t| - 4a          for 1 < |t| < 2
         = 0                                       otherwise
    """
    t = np.abs(t)
    t2, t3 = t * t, t * t * t
    near = (a + 2) * t3 - (a + 3) * t2 + 1
    far = a * t3 - 5 * a * t2 + 8 * a * t - 4 * a
    return np.where(t <= 1, near, np.where(t < 2, far, 0.0))


def lanczos3_kernel(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.where(np.abs(t) < 3, np.sinc(t) * np.sinc(t / 3.0), 0.0)


KERNELS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], float]] = {
    "bilinear": (triangle_kernel, 1.0),
    "bicubic": (keys_cubic_kernel, 2.0),
    "lanczos3": (lanczos3_kernel, 3.0),
}


# ---------------------------------------------------------- weight matrices
@lru_cache(maxsize=64)
def weight_matrix(in_size: int, out_size: int, method: str) -> sp.csr_matrix:
    """Sparse (out_size x in_size) matrix whose rows are interpolation weights."""
    if in_size <= 0 or out_size <= 0:
        raise ValueError("Sizes must be positive")
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}")
    scale = out_size / in_size
    centres = (np.arange(out_size) + 0.5) / scale - 0.5  # input coordinates

    if method == "nearest":
        # Pillow convention: src = floor((i + 0.5) / s)
        idx = np.clip(np.floor((np.arange(out_size) + 0.5) / scale).astype(np.int64), 0, in_size - 1)
        return sp.csr_matrix((np.ones(out_size), (np.arange(out_size), idx)), shape=(out_size, in_size))

    kernel, support = KERNELS[method]
    stretch = max(1.0, 1.0 / scale)          # anti-alias when downsampling
    radius = support * stretch
    lo = np.floor(centres - radius).astype(np.int64) + 1
    taps = int(np.ceil(2 * radius)) + 1
    j = lo[:, None] + np.arange(taps)[None, :]              # (out, taps)
    w = kernel((j - centres[:, None]) / stretch)
    valid = (j >= 0) & (j < in_size)
    w = np.where(valid, w, 0.0)
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / np.where(np.abs(w_sum) < 1e-12, 1.0, w_sum)
    rows = np.repeat(np.arange(out_size), taps)
    mask = valid.ravel() & (w.ravel() != 0)
    return sp.csr_matrix((w.ravel()[mask], (rows[mask], j.ravel()[mask])), shape=(out_size, in_size))


# ------------------------------------------------------------------- resize
def resize(img: np.ndarray, out_h: int, out_w: int, method: str = "bicubic",
           clip: bool = True) -> np.ndarray:
    """Resize an (H, W) or (H, W, C) float image to (out_h, out_w).

    ``clip=False`` keeps the bicubic/Lanczos overshoot (useful to measure ringing).
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected (H, W) or (H, W, C) array, got shape {img.shape}")
    squeeze = img.ndim == 2
    x = img[..., None] if squeeze else img
    x = x.astype(np.float32, copy=False)
    h, w, c = x.shape
    wh = weight_matrix(h, out_h, method).astype(np.float32)
    ww = weight_matrix(w, out_w, method).astype(np.float32)
    tmp = wh @ x.reshape(h, w * c)                                   # (out_h, w*c)
    tmp = tmp.reshape(out_h, w, c).transpose(1, 0, 2).reshape(w, out_h * c)
    out = (ww @ tmp).reshape(out_w, out_h, c).transpose(1, 0, 2)    # (out_h, out_w, c)
    out = np.ascontiguousarray(out, dtype=np.float32)
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out[..., 0] if squeeze else out


def upscale(img: np.ndarray, scale: float, method: str = "bicubic", clip: bool = True) -> np.ndarray:
    """Upscale by ``scale`` per dimension (output size = round(H*scale), round(W*scale))."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    h, w = img.shape[:2]
    return resize(img, int(round(h * scale)), int(round(w * scale)), method, clip)


def downscale(img: np.ndarray, scale: int, method: str = "bicubic") -> np.ndarray:
    """Anti-aliased downscale by an integer factor (the classic SR degradation).

    The image must be divisible by ``scale``; use :func:`modcrop` first.
    """
    h, w = img.shape[:2]
    if h % scale or w % scale:
        raise ValueError(f"Image {h}x{w} not divisible by scale {scale}; call modcrop first")
    return resize(img, h // scale, w // scale, method)


def modcrop(img: np.ndarray, scale: int) -> np.ndarray:
    """Crop bottom/right so that H and W are multiples of ``scale``."""
    h, w = img.shape[:2]
    return img[: h - h % scale, : w - w % scale]
