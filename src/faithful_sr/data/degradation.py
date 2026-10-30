r"""Configurable LR-image degradation model.

We model the observed low-resolution image as

.. math::  y = \mathrm{JPEG}_q\big( (x * k_\sigma)\downarrow_s + n \big),
           \qquad n \sim \mathcal N(0, \sigma_n^2)

applied in this fixed, recorded order:

1. optional isotropic Gaussian blur ``k_sigma`` on the HR image,
2. anti-aliased bicubic downsampling by the integer factor ``s`` (our own
   implementation in :mod:`faithful_sr.models.interpolation`),
3. optional additive Gaussian noise on the LR image,
4. 8-bit quantisation,
5. optional JPEG compression (Pillow encoder; 4:4:4 chroma, quality ``q``).

The blur kernel is symmetric and centred and the resampler uses the
pixel-centre convention, so LR and HR stay exactly aligned and paired
full-reference metrics remain valid for every preset.

Randomness is fully determined by ``(seed, image_id, variant)`` so evaluation
LR images can be regenerated bit-exactly from the manifest + config.
"""
from __future__ import annotations

import io
import zlib
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d

from faithful_sr.models.interpolation import downscale
from faithful_sr.utils.image_io import quantize, to_float, to_uint8


@dataclass
class OpConfig:
    prob: float = 0.0          # probability the op is applied
    low: float = 0.0           # parameter range [low, high]
    high: float = 0.0

    def validate(self, name: str, lo_lim: float, hi_lim: float) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"{name}.prob must be in [0, 1]")
        if self.prob > 0 and not (lo_lim <= self.low <= self.high <= hi_lim):
            raise ValueError(f"{name} range must satisfy {lo_lim} <= low <= high <= {hi_lim}")


@dataclass
class DegradationConfig:
    """Parameters of the degradation model (all values documented in configs/degradation.yaml)."""

    name: str = "bicubic"
    blur_sigma: OpConfig = field(default_factory=OpConfig)      # HR pixels
    noise_sigma: OpConfig = field(default_factory=OpConfig)     # in /255 units
    jpeg_quality: OpConfig = field(default_factory=OpConfig)    # 1..100

    def validate(self) -> "DegradationConfig":
        self.blur_sigma.validate("blur_sigma", 0.0, 5.0)
        self.noise_sigma.validate("noise_sigma", 0.0, 50.0)
        self.jpeg_quality.validate("jpeg_quality", 1, 100)
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "DegradationConfig":
        kw = {"name": d.get("name", "custom")}
        for k in ("blur_sigma", "noise_sigma", "jpeg_quality"):
            if k in d and d[k] is not None:
                kw[k] = OpConfig(**d[k])
        return cls(**kw).validate()

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, DegradationConfig] = {
    # Classic benchmark setting: bicubic downsampling only.
    "bicubic": DegradationConfig("bicubic"),
    # Single-factor presets (used to isolate each effect).
    "blur": DegradationConfig("blur", blur_sigma=OpConfig(1.0, 0.4, 1.6)),
    "noise": DegradationConfig("noise", noise_sigma=OpConfig(1.0, 1.0, 8.0)),
    "jpeg": DegradationConfig("jpeg", jpeg_quality=OpConfig(1.0, 50, 90)),
    # Mild combined pipeline: each op applied with probability p.
    "mixed": DegradationConfig("mixed", blur_sigma=OpConfig(0.7, 0.4, 1.6),
                               noise_sigma=OpConfig(0.5, 1.0, 6.0),
                               jpeg_quality=OpConfig(0.7, 55, 95)),
}


def get_preset(name_or_dict) -> DegradationConfig:
    if isinstance(name_or_dict, DegradationConfig):
        return name_or_dict.validate()
    if isinstance(name_or_dict, dict):
        return DegradationConfig.from_dict(name_or_dict)
    if name_or_dict not in PRESETS:
        raise ValueError(f"Unknown degradation '{name_or_dict}'. Presets: {list(PRESETS)}")
    return PRESETS[name_or_dict]


def image_rng(seed: int, image_id: str, variant: int = 0) -> np.random.Generator:
    """Deterministic generator for one (image, variant) pair."""
    return np.random.default_rng([int(seed), zlib.crc32(image_id.encode()), int(variant)])


# --------------------------------------------------------------- operators
def gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (alignment-preserving)."""
    if sigma <= 0:
        return img
    k = gaussian_kernel1d(sigma)
    out = convolve1d(img, k, axis=0, mode="reflect")
    return convolve1d(out, k, axis=1, mode="reflect")


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(to_uint8(img), "RGB").save(buf, format="JPEG", quality=int(quality), subsampling=0)
    buf.seek(0)
    return to_float(np.asarray(Image.open(buf).convert("RGB")))


def degrade(hr: np.ndarray, scale: int, cfg: DegradationConfig | str = "bicubic",
            rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, dict]:
    """Create an LR image from an HR float image (H and W must be divisible by ``scale``).

    Returns ``(lr, params)`` where ``params`` records exactly what was applied.
    """
    cfg = get_preset(cfg)
    rng = rng or np.random.default_rng(0)
    params: dict = {"preset": cfg.name, "scale": scale,
                    "order": ["blur", "bicubic_down", "noise", "quantize_8bit", "jpeg"]}
    x = hr.astype(np.float32, copy=False)

    # Draw every random number in a fixed order so outcomes are reproducible
    # regardless of which ops end up enabled.
    u = rng.random(3)
    blur_s = rng.uniform(cfg.blur_sigma.low, cfg.blur_sigma.high) if cfg.blur_sigma.high > 0 else 0.0
    noise_s = rng.uniform(cfg.noise_sigma.low, cfg.noise_sigma.high) if cfg.noise_sigma.high > 0 else 0.0
    q = int(round(rng.uniform(cfg.jpeg_quality.low, cfg.jpeg_quality.high))) if cfg.jpeg_quality.high > 0 else 100
    noise_seed = int(rng.integers(0, 2**31 - 1))

    if u[0] < cfg.blur_sigma.prob and blur_s > 0:
        x = gaussian_blur(x, blur_s)
        params["blur_sigma"] = round(float(blur_s), 4)
    x = downscale(x, scale, "bicubic")
    if u[1] < cfg.noise_sigma.prob and noise_s > 0:
        n = np.random.default_rng(noise_seed).normal(0.0, noise_s / 255.0, x.shape).astype(np.float32)
        x = np.clip(x + n, 0.0, 1.0)
        params["noise_sigma_255"] = round(float(noise_s), 4)
    x = quantize(x)
    if u[2] < cfg.jpeg_quality.prob:
        x = jpeg_compress(x, q)
        params["jpeg_quality"] = q
    return x, params
