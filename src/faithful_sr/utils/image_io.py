"""Image I/O and format conversion utilities.

Conventions used throughout the project
---------------------------------------
* In-memory images are ``numpy.float32`` arrays of shape ``(H, W, 3)``, RGB,
  values in ``[0, 1]``.  Call these "float images".
* Tensors are ``torch.float32`` of shape ``(N, 3, H, W)`` in ``[0, 1]``.
* Files are read with Pillow, EXIF orientation is applied (so what you see in an
  image viewer is what the model sees), and colour mode is normalised to RGB.
* Alpha is *not* silently discarded: :func:`load_image` returns it separately so
  callers can upscale it with a classical method and re-attach it on save.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Refuse absurdly large inputs instead of crashing the machine (~100 MP).
MAX_INPUT_PIXELS = 100_000_000


class ImageLoadError(RuntimeError):
    """Raised for missing, corrupt, unsupported or oversized image files."""


@dataclass
class LoadedImage:
    """An RGB float image plus everything we learned while loading it."""

    rgb: np.ndarray                      # (H, W, 3) float32 in [0, 1]
    alpha: Optional[np.ndarray] = None   # (H, W) float32 in [0, 1] or None
    original_mode: str = "RGB"
    path: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8/uint16/float array -> float32 in [0, 1] (no colour changes)."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img.astype(np.float32), 0.0, 1.0)
    raise TypeError(f"Unsupported image dtype {img.dtype}")


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float image in [0, 1] -> uint8 with round-half-up quantisation."""
    if img.dtype == np.uint8:
        return img
    return np.clip(np.floor(img * 255.0 + 0.5), 0, 255).astype(np.uint8)


def quantize(img: np.ndarray) -> np.ndarray:
    """Simulate 8-bit storage of a float image (used for LR inputs)."""
    return to_float(to_uint8(img))


def pil_to_float(pil: Image.Image) -> LoadedImage:
    """Convert a PIL image (any mode) to a :class:`LoadedImage`."""
    warnings: list[str] = []
    mode = pil.mode
    alpha = None
    if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in pil.info):
        rgba = pil.convert("RGBA")
        a = np.asarray(rgba)[..., 3]
        if a.min() < 255:
            alpha = a.astype(np.float32) / 255.0
            warnings.append("Image has an alpha channel; it is upscaled separately with bicubic interpolation.")
        rgb = np.asarray(rgba.convert("RGB"))
    elif mode in ("L", "1", "I;16", "I"):
        if mode in ("I;16", "I"):
            arr = np.asarray(pil).astype(np.float32)
            arr = arr / (65535.0 if arr.max() > 255 else 255.0)
            rgb = np.repeat(np.clip(arr, 0, 1)[..., None], 3, axis=2)
            return LoadedImage(rgb.astype(np.float32), None, mode, None,
                               ["Grayscale 16-bit input replicated to 3 channels."])
        rgb = np.asarray(pil.convert("RGB"))
        warnings.append("Grayscale input replicated to 3 RGB channels.")
    elif mode == "CMYK":
        rgb = np.asarray(pil.convert("RGB"))
        warnings.append("CMYK input converted to RGB with Pillow's default transform; colours may shift slightly.")
    else:
        rgb = np.asarray(pil.convert("RGB"))
    return LoadedImage(to_float(rgb), alpha, mode, None, warnings)


def load_image(path: str | Path) -> LoadedImage:
    """Load an image file as RGB float32, applying EXIF orientation.

    Raises :class:`ImageLoadError` with a human-readable message on failure.
    """
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"File not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported file type '{p.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    try:
        with Image.open(p) as pil:
            if pil.width * pil.height > MAX_INPUT_PIXELS:
                raise ImageLoadError(
                    f"Image is {pil.width}x{pil.height}; inputs above {MAX_INPUT_PIXELS:,} px are refused.")
            pil.load()
            pil = ImageOps.exif_transpose(pil)
            loaded = pil_to_float(pil)
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ImageLoadError(f"Could not read image {p.name}: file may be corrupt ({e}).") from e
    loaded.path = str(p)
    return loaded


def save_image(img: np.ndarray, path: str | Path, alpha: Optional[np.ndarray] = None,
               jpeg_quality: int = 95, overwrite_protect: Optional[str | Path] = None) -> Path:
    """Save a float (or uint8) RGB image. PNG by default (lossless).

    ``overwrite_protect``: path of the original input; saving onto it is refused.
    """
    p = Path(path)
    if overwrite_protect is not None and p.resolve() == Path(overwrite_protect).resolve():
        raise ValueError("Refusing to overwrite the original input image.")
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = to_uint8(img)
    if alpha is not None and p.suffix.lower() in (".png", ".webp", ".tif", ".tiff"):
        a = to_uint8(alpha)[..., None]
        pil = Image.fromarray(np.concatenate([arr, a], axis=2), "RGBA")
    else:
        pil = Image.fromarray(arr, "RGB")
    if p.suffix.lower() in (".jpg", ".jpeg"):
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        pil.save(p, quality=int(jpeg_quality), subsampling=0)
    else:
        pil.save(p)
    return p


# ---------------------------------------------------------------- torch glue
def float_to_tensor(img: np.ndarray):
    """(H, W, 3) float32 -> (1, 3, H, W) torch tensor (shares no memory)."""
    import torch
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got {img.shape}")
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).float()


def tensor_to_float(t) -> np.ndarray:
    """(1, 3, H, W) or (3, H, W) tensor -> (H, W, 3) float32 clipped to [0, 1]."""
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError("tensor_to_float expects a batch of size 1")
        t = t[0]
    arr = t.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return np.ascontiguousarray(arr, dtype=np.float32)
