"""Shared inference API used by the CLI, the evaluation code and the GUI.

Every method is a callable ``f(img, scale) -> img`` on float RGB images
(``(H, W, 3)`` float32 in ``[0, 1]``).  :func:`upscale_image` adds validation,
alpha handling, timing and metadata so all front-ends behave identically.
"""
from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from faithful_sr.models import interpolation as interp
from faithful_sr.utils.image_io import LoadedImage, load_image, save_image

UpscaleFn = Callable[[np.ndarray, int], np.ndarray]

CLASSICAL = ("nearest", "bilinear", "bicubic")
SUPPORTED_SCALES = (2, 3, 4, 6)
HALLUCINATION_WARNING = (
    "Learned super-resolution estimates missing detail; it cannot guarantee recovery of the "
    "original scene. Fine text, faces and textures may be altered.")


@dataclass
class UpscaleResult:
    image: np.ndarray
    alpha: Optional[np.ndarray]
    method: str
    scale: int
    input_size: tuple[int, int]      # (W, H)
    output_size: tuple[int, int]     # (W, H)
    runtime_s: float
    device: str
    warnings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("image")
        d.pop("alpha")
        d["has_alpha"] = self.alpha is not None
        d["python"] = platform.python_version()
        return d


# ---------------------------------------------------------------- registry
_REGISTRY: dict[str, Callable[..., UpscaleFn]] = {}


def register(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        return factory
    return deco


for _m in CLASSICAL:
    def _factory(_m=_m, **_):
        return lambda img, s: interp.upscale(img, s, _m)
    register(_m)(_factory)


def available_methods() -> list[str]:
    """Methods that can be constructed (learned models register themselves on import)."""
    _import_optional()
    return list(_REGISTRY)


def _import_optional() -> None:
    # Learned methods live in modules that need torch; import lazily so the
    # classical baselines work in a NumPy-only environment.
    try:
        import faithful_sr.inference.learned  # noqa: F401  (registers cnn methods)
    except ImportError:
        pass


def get_method(name: str, **kwargs) -> UpscaleFn:
    _import_optional()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def _device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return f"cpu:{platform.processor() or platform.machine()}"


def upscale_array(img: np.ndarray, method: str, scale: int, alpha: Optional[np.ndarray] = None,
                  fn: Optional[UpscaleFn] = None, **method_kwargs) -> UpscaleResult:
    """Upscale an in-memory float RGB image."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got {img.shape}")
    if int(scale) != scale or scale not in SUPPORTED_SCALES:
        raise ValueError(f"Unsupported scale {scale}; supported: {SUPPORTED_SCALES}")
    scale = int(scale)
    fn = fn or get_method(method, **method_kwargs)
    h, w = img.shape[:2]
    t0 = time.perf_counter()
    out = fn(img, scale)
    runtime = time.perf_counter() - t0
    expected = (h * scale, w * scale)
    if out.shape[:2] != expected:
        raise RuntimeError(f"{method} produced {out.shape[:2]}, expected {expected}")
    warnings: list[str] = []
    if method not in CLASSICAL:
        warnings.append(HALLUCINATION_WARNING)
    out_alpha = interp.upscale(alpha, scale, "bicubic") if alpha is not None else None
    return UpscaleResult(out, out_alpha, method, scale, (w, h), (w * scale, h * scale),
                         runtime, _device_name() if method not in CLASSICAL else "cpu:numpy", warnings)


def upscale_file(path: str | Path, method: str, scale: int, output: Optional[str | Path] = None,
                 **method_kwargs) -> UpscaleResult:
    """Load, upscale and optionally save an image file (never overwrites the input)."""
    loaded: LoadedImage = load_image(path)
    res = upscale_array(loaded.rgb, method, scale, loaded.alpha, **method_kwargs)
    res.warnings = loaded.warnings + res.warnings
    if output is not None:
        save_image(res.image, output, res.alpha, overwrite_protect=path)
        res.extra["output_path"] = str(output)
    return res
