"""Prepare manifest, splits, training sub-images and fixed evaluation LR/HR pairs.

python scripts/prepare_dataset.py --config configs/dataset.yaml
python scripts/prepare_dataset.py --config configs/sample_dataset.yaml   # tiny synthetic sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faithful_sr.data.prepare import prepare  # noqa: E402
from faithful_sr.utils.config import deep_update, load_yaml, parse_overrides  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--root", default=str(ROOT), help="project root that config paths are relative to")
    ap.add_argument("--degradation-config", default="configs/degradation.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value overrides, e.g. scales=[3]")
    a = ap.parse_args(argv)
    root = Path(a.root)
    cfg = deep_update(load_yaml(root / a.config), parse_overrides(a.overrides))
    dpath = root / a.degradation_config
    if dpath.exists():
        cfg.setdefault("degradations", load_yaml(dpath))
    summary = prepare(cfg, root)
    print("split counts:", summary["split_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
