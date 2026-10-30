r"""Full-reference image-quality metrics (implemented from their definitions).

Conventions (identical for every method; standard in SR literature)
------------------------------------------------------------------
* Inputs: float RGB images in [0, 1], same shape, exactly aligned.
* ``crop_border = scale`` pixels are removed from each side before scoring
  (edge pixels depend on the padding convention of each method).
* Y channel = ITU-R BT.601 luma as in MATLAB ``rgb2ycbcr``:
  ``Y = (16 + 65.481 R + 128.553 G + 24.966 B) / 255``  (R, G, B in [0, 1]).
* PSNR = 10 log10(1 / MSE)  with peak 1 (equivalently 255 on 8-bit data).
* SSIM (Wang et al., 2004): Gaussian window sigma = 1.5 (11x11), K1 = 0.01,
  K2 = 0.03, L = 1, population covariances; mean over the 'valid' region.
  Matches ``skimage.metrics.structural_similarity(gaussian_weights=True,
  sigma=1.5, use_sample_covariance=False, data_range=1)`` (unit-tested).
* LPIPS (optional, Zhang et al., 2018; AlexNet) on RGB in [-1, 1]: lower = more
  perceptually similar.  It is a learned perceptual distance, not a proof that
  details are authentic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    return (16.0 + 65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0


def _prep(pred: np.ndarray, gt: np.ndarray, crop_border: int, y_only: bool):
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if crop_border:
        pred = pred[crop_border:-crop_border, crop_border:-crop_border]
        gt = gt[crop_border:-crop_border, crop_border:-crop_border]
    if y_only and pred.ndim == 3:
        return rgb_to_y(pred), rgb_to_y(gt)
    return pred.astype(np.float64), gt.astype(np.float64)


def psnr(pred: np.ndarray, gt: np.ndarray, crop_border: int = 0, y_only: bool = True) -> float:
    p, g = _prep(pred, gt, crop_border, y_only)
    mse = float(np.mean((p - g) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def _ssim_single(x: np.ndarray, y: np.ndarray, sigma: float = 1.5) -> float:
    c1, c2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    f = lambda a: gaussian_filter(a, sigma, truncate=3.5, mode="reflect")  # noqa: E731
    mx, my = f(x), f(y)
    vx = f(x * x) - mx * mx
    vy = f(y * y) - my * my
    cxy = f(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    pad = int(3.5 * sigma + 0.5)          # = 5 -> 11x11 window; ignore the padded border
    return float(s[pad:-pad, pad:-pad].mean())


def ssim(pred: np.ndarray, gt: np.ndarray, crop_border: int = 0, y_only: bool = True) -> float:
    p, g = _prep(pred, gt, crop_border, y_only)
    if p.ndim == 2:
        return _ssim_single(p, g)
    return float(np.mean([_ssim_single(p[..., c], g[..., c]) for c in range(p.shape[2])]))


class LPIPSMetric:
    """Lazy wrapper around the ``lpips`` package (optional dependency)."""

    def __init__(self, net: str = "alex", device: Optional[str] = None):
        import lpips  # raises ImportError if not installed
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()

    def __call__(self, pred: np.ndarray, gt: np.ndarray, crop_border: int = 0) -> float:
        if crop_border:
            pred = pred[crop_border:-crop_border, crop_border:-crop_border]
            gt = gt[crop_border:-crop_border, crop_border:-crop_border]
        t = lambda a: self.torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].float().to(self.device) * 2 - 1  # noqa: E731
        with self.torch.no_grad():
            return float(self.model(t(pred), t(gt)).item())


def try_lpips(device: Optional[str] = None) -> Optional[LPIPSMetric]:
    try:
        return LPIPSMetric(device=device)
    except Exception:
        return None


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of the Y channel (used for edge/ringing analysis)."""
    from scipy.ndimage import sobel
    y = rgb_to_y(img) if img.ndim == 3 else img.astype(np.float64)
    return np.hypot(sobel(y, axis=0), sobel(y, axis=1))


def edge_psnr(pred: np.ndarray, gt: np.ndarray, crop_border: int = 0, quantile: float = 0.9) -> float:
    """PSNR restricted to the strongest-gradient pixels of the ground truth (edges, lines, text)."""
    p, g = _prep(pred, gt, crop_border, True)
    gm = gradient_magnitude(g)
    mask = gm >= np.quantile(gm, quantile)
    mse = float(np.mean((p[mask] - g[mask]) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def overshoot(pred: np.ndarray, gt: np.ndarray, crop_border: int = 0) -> float:
    """Ringing indicator: mean amount by which pred exceeds the local GT [min, max] range (3x3)."""
    from scipy.ndimage import maximum_filter, minimum_filter
    p, g = _prep(pred, gt, crop_border, True)
    hi, lo = maximum_filter(g, 3), minimum_filter(g, 3)
    return float(np.mean(np.maximum(p - hi, 0) + np.maximum(lo - p, 0)))
