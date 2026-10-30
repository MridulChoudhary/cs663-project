"""Config loading, run metadata and seeding helpers."""
from __future__ import annotations

import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def parse_overrides(pairs: list[str]) -> dict:
    """``["train.lr=1e-4", "model.depth=8"]`` -> nested dict (values parsed as YAML)."""
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must look like key.sub=value")
        key, val = pair.split("=", 1)
        d = out
        parts = key.split(".")
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).resolve().parent).decode().strip()
    except Exception:
        return "unknown"


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "git_commit": _git_commit(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p
