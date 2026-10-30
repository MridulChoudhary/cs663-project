"""Checkpoint save / load / resume (never deletes earlier checkpoints)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def save_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None, *, iteration: int,
                    epoch: int, best_metric: float, config: dict, extra: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(), "iteration": iteration, "epoch": epoch, "best_metric": best_metric,
        "config": config, "model_config": config.get("model", {}),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra:
        state.update(extra)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)          # atomic: a crash never leaves a half-written checkpoint
    return path


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model first (scripts/train_model.py) "
            f"or pass the correct --checkpoint path.")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Could not read checkpoint {path}: {e}") from e


def restore_rng(state: dict) -> None:
    rng = state.get("rng")
    if not rng:
        return
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def find_resume(output_dir: Path) -> Optional[Path]:
    p = Path(output_dir) / "latest.pt"
    return p if p.exists() else None
