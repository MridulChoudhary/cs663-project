"""Train an SR CNN.

python scripts/train_model.py --config configs/train_fsrnet_x3.yaml
python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto     # continue after interruption
python scripts/train_model.py --config configs/smoke.yaml                             # 1-2 minute CPU smoke test
python scripts/train_model.py --config configs/train_fsrnet_x3.yaml train.loss=mse output_dir=outputs/runs/fsrnet_mse
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faithful_sr.training.train import train  # noqa: E402
from faithful_sr.utils.config import deep_update, load_yaml, parse_overrides  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="'auto' (use output_dir/latest.pt) or a checkpoint path")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    a = ap.parse_args(argv)
    cfg = deep_update(load_yaml(a.config), parse_overrides(a.overrides))
    res = train(cfg, a.resume)
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
