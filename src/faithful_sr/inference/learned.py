"""Learned upscalers: our trained CNNs and the pretrained Real-ESRGAN comparator.

Importing this module registers two methods with the shared inference API:

``cnn``  (kwargs: ``checkpoint``, ``tile``, ``overlap``, ``six_strategy``, ``device``)
    Any checkpoint produced by ``scripts/train_model.py`` (FSRNet or SRCNN).
    * requested scale == model scale -> direct inference
    * 6x with a 3x model -> staged, chosen by ``six_strategy``:
        - ``"cnn_then_bicubic"``  : CNN 3x, then bicubic 2x            (default)
        - ``"cnn_twice_then_down"``: CNN 3x, CNN 3x (=9x), anti-aliased bicubic down to 6x
      Errors made by the first stage propagate into the second.
    * any other mismatch raises a clear error (never silently run at an unsupported scale).

``realesrgan``  (kwargs: ``weights``, ``tile``, ``overlap``, ``device``)
    Real-ESRGAN x4plus (Wang et al., ICCVW 2021; BSD-3-Clause), loaded with
    ``spandrel`` from an explicitly downloaded ``.pth`` (``scripts/download_weights.py``).
    Native scale is 4x.  Adaptation (documented in docs/methodology.md):
        3x = 4x output, anti-aliased bicubic downscale by 3/4
        6x = 4x output, bicubic upscale by 3/2
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from faithful_sr.inference.tiling import tiled_upscale
from faithful_sr.inference.upscale import register
from faithful_sr.models import interpolation as interp
from faithful_sr.models.cnn import build_model
from faithful_sr.training.checkpointing import load_checkpoint
from faithful_sr.utils.image_io import float_to_tensor, tensor_to_float

DEFAULT_REALESRGAN = Path("models/pretrained/RealESRGAN_x4plus.pth")


def _device(pref: Optional[str] = None) -> torch.device:
    if pref and pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_torch(model, img: np.ndarray, device) -> np.ndarray:
    try:
        with torch.no_grad():
            return tensor_to_float(model(float_to_tensor(img).to(device)))
    except torch.OutOfMemoryError as e:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError("Out of GPU memory - enable tiling (e.g. tile=256) or use a smaller tile size.") from e
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "DefaultCPUAllocator" in str(e):
            raise RuntimeError("Out of memory - enable tiling (e.g. tile=256) or use a smaller tile size.") from e
        raise


@lru_cache(maxsize=8)
def load_cnn(checkpoint: str, device: str):
    st = load_checkpoint(Path(checkpoint), map_location=device)
    model = build_model(st["model_config"]).to(device).eval()
    model.load_state_dict(st["model"])
    return model


@register("cnn")
def cnn_factory(checkpoint: Optional[str] = None, tile: int = 0, overlap: Optional[int] = None,
                six_strategy: str = "cnn_then_bicubic", device: Optional[str] = None, **_):
    if not checkpoint:
        raise FileNotFoundError("The 'cnn' method needs a checkpoint (e.g. models/fsrnet_x3_l1_bicubic.pt). "
                                "Train one with scripts/train_model.py or download it from the GitHub release.")
    dev = _device(device)
    model = load_cnn(str(checkpoint), str(dev))
    ms = model.scale
    ov = model.receptive_field_lr() // 2 + 2 if overlap is None else overlap

    def native(img: np.ndarray) -> np.ndarray:
        return tiled_upscale(img, lambda t: _run_torch(model, t, dev), ms, tile, ov)

    def fn(img: np.ndarray, scale: int) -> np.ndarray:
        if scale == ms:
            return native(img)
        if scale == 6 and ms == 3:
            if six_strategy == "cnn_then_bicubic":
                return interp.upscale(native(img), 2, "bicubic")
            if six_strategy == "cnn_twice_then_down":
                x9 = native(native(img))
                h, w = img.shape[:2]
                return interp.resize(x9, h * 6, w * 6, "bicubic")
            raise ValueError(f"Unknown six_strategy '{six_strategy}'")
        raise ValueError(f"This checkpoint is a {ms}x model; {scale}x is not supported "
                         f"(supported: {ms}x" + (", 6x staged)" if ms == 3 else ")"))
    fn.model_scale = ms
    return fn


@lru_cache(maxsize=2)
def load_realesrgan(weights: str, device: str):
    try:
        import spandrel
    except ImportError as e:
        raise ImportError("Real-ESRGAN support needs the optional dependency: pip install -r requirements-pretrained.txt") from e
    p = Path(weights)
    if not p.exists():
        raise FileNotFoundError(f"Real-ESRGAN weights not found at {p}. Run: python scripts/download_weights.py "
                                f"(downloads RealESRGAN_x4plus.pth, ~64 MB, BSD-3-Clause).")
    desc = spandrel.ModelLoader().load_from_file(str(p))
    return desc.model.to(device).eval(), int(desc.scale)


@register("realesrgan")
def realesrgan_factory(weights: Optional[str] = None, tile: int = 0, overlap: int = 16,
                       device: Optional[str] = None, **_):
    dev = _device(device)
    model, ns = load_realesrgan(str(weights or DEFAULT_REALESRGAN), str(dev))

    def native(img: np.ndarray) -> np.ndarray:
        return tiled_upscale(img, lambda t: _run_torch(model, t, dev), ns, tile, overlap)

    def fn(img: np.ndarray, scale: int) -> np.ndarray:
        h, w = img.shape[:2]
        out = native(img)
        if scale == ns:
            return out
        return interp.resize(out, h * scale, w * scale, "bicubic")   # anti-aliased when shrinking
    fn.model_scale = ns
    return fn
