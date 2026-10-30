"""PyTorch datasets.

* :class:`TrainPatchDataset` — random aligned LR/HR patches from training
  sub-images, degraded *on the fly* (a fresh degradation sample per patch), with
  optional flips / 90° rotations.
* :class:`PairedImageDataset` — fixed, pre-generated LR/HR image pairs (val/test),
  so evaluation is bit-exactly reproducible.

Randomness in the training set is a deterministic function of
``(seed, epoch, index)``; the trainer calls :meth:`TrainPatchDataset.set_epoch`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from faithful_sr.data.degradation import degrade, get_preset
from faithful_sr.utils.image_io import SUPPORTED_EXTENSIONS, load_image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # allow NumPy-only use of the rest of the package
    torch = None
    Dataset = object


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def augment(hr: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    """``mode``: 'none' | 'flips' | 'flips_rot' (dihedral group D4)."""
    if mode == "none":
        return hr
    if rng.random() < 0.5:
        hr = hr[:, ::-1]
    if rng.random() < 0.5:
        hr = hr[::-1]
    if mode == "flips_rot" and rng.random() < 0.5:
        hr = hr.transpose(1, 0, 2)
    return np.ascontiguousarray(hr)


def extract_subimages(hr: np.ndarray, size: int, step: int) -> list[tuple[int, int, np.ndarray]]:
    """Tile an HR image into ``size``-square crops with stride ``step`` (edges included)."""
    h, w = hr.shape[:2]
    if h < size or w < size:
        return [(0, 0, hr)]
    ys = list(range(0, h - size + 1, step))
    xs = list(range(0, w - size + 1, step))
    if ys[-1] != h - size:
        ys.append(h - size)
    if xs[-1] != w - size:
        xs.append(w - size)
    return [(y, x, hr[y:y + size, x:x + size]) for y in ys for x in xs]


class TrainPatchDataset(Dataset):
    def __init__(self, folder: str | Path, scale: int, lr_patch: int = 48,
                 degradation="bicubic", augment_mode: str = "flips_rot", seed: int = 663,
                 repeat: int = 1, cache: bool = False, files: Optional[Sequence[Path]] = None):
        self.files = list(files) if files is not None else list_images(Path(folder))
        if not self.files:
            raise RuntimeError(f"No training images found in {folder}. Run scripts/prepare_dataset.py first.")
        self.scale, self.lr_patch = int(scale), int(lr_patch)
        self.hr_patch = self.lr_patch * self.scale
        self.deg = get_preset(degradation)
        self.augment_mode, self.seed, self.repeat = augment_mode, int(seed), int(repeat)
        self.epoch = 0
        self._cache: dict[int, np.ndarray] = {} if cache else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.files) * self.repeat

    def _load(self, k: int) -> np.ndarray:
        if self._cache is not None and k in self._cache:
            return self._cache[k]
        img = load_image(self.files[k]).rgb
        if self._cache is not None:
            self._cache[k] = img
        return img

    def __getitem__(self, i: int):
        k = i % len(self.files)
        rng = np.random.default_rng([self.seed, self.epoch, i])
        hr = self._load(k)
        h, w = hr.shape[:2]
        P = self.hr_patch
        if h < P or w < P:
            pad = ((0, max(0, P - h)), (0, max(0, P - w)), (0, 0))
            hr = np.pad(hr, pad, mode="reflect")
            h, w = hr.shape[:2]
        # crop on the LR grid so the HR patch is aligned with the LR patch
        y = int(rng.integers(0, (h - P) // self.scale + 1)) * self.scale
        x = int(rng.integers(0, (w - P) // self.scale + 1)) * self.scale
        hr_p = augment(hr[y:y + P, x:x + P], rng, self.augment_mode)
        lr_p, _ = degrade(hr_p, self.scale, self.deg, rng)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))  # noqa: E731
        return to_t(lr_p), to_t(hr_p)


class PairedImageDataset(Dataset):
    """Pre-generated pairs: ``lr_dir/<id>.png`` <-> ``hr_dir/<id>.png``.

    ``center_crop_hr``: optionally evaluate on an HR centre crop (fast validation).
    """

    def __init__(self, lr_dir: str | Path, hr_dir: str | Path, scale: int,
                 center_crop_hr: int = 0, ids: Optional[Sequence[str]] = None):
        self.lr_dir, self.hr_dir, self.scale = Path(lr_dir), Path(hr_dir), int(scale)
        lr_files = {p.stem: p for p in list_images(self.lr_dir)} if self.lr_dir.exists() else {}
        self.ids = [i for i in (ids or sorted(lr_files)) if i in lr_files and (self.hr_dir / f"{i}.png").exists()]
        if not self.ids:
            raise RuntimeError(f"No LR/HR pairs found in {lr_dir} / {hr_dir}")
        self.lr_files = lr_files
        self.crop = int(center_crop_hr) - int(center_crop_hr) % self.scale

    def __len__(self) -> int:
        return len(self.ids)

    def get_numpy(self, i: int) -> tuple[str, np.ndarray, np.ndarray]:
        iid = self.ids[i]
        lr = load_image(self.lr_files[iid]).rgb
        hr = load_image(self.hr_dir / f"{iid}.png").rgb
        if hr.shape[0] != lr.shape[0] * self.scale or hr.shape[1] != lr.shape[1] * self.scale:
            raise RuntimeError(f"{iid}: HR {hr.shape[:2]} is not {self.scale}x LR {lr.shape[:2]}")
        if self.crop:
            c, s = self.crop, self.scale
            cl = c // s
            y = max(0, (lr.shape[0] - cl) // 2)
            x = max(0, (lr.shape[1] - cl) // 2)
            lr = lr[y:y + cl, x:x + cl]
            hr = hr[y * s:(y + cl) * s, x * s:(x + cl) * s]
        return iid, lr, hr

    def __getitem__(self, i: int):
        iid, lr, hr = self.get_numpy(i)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))  # noqa: E731
        return to_t(lr), to_t(hr), iid
