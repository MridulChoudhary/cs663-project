"""Evaluate all configured methods and build the report.

python scripts/evaluate_model.py --config configs/eval.yaml                 # full benchmark + report
python scripts/evaluate_model.py --config configs/eval.yaml --runtime       # also E8 runtime table
python scripts/evaluate_model.py --config configs/eval.yaml --real-world    # also qualitative real-world grids
python scripts/evaluate_model.py --report-only outputs/eval/main            # rebuild plots/tables only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faithful_sr.evaluation.benchmark import run_benchmark, run_real_world, run_runtime  # noqa: E402
from faithful_sr.evaluation.report import build_report  # noqa: E402
from faithful_sr.utils.config import deep_update, load_yaml, parse_overrides  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--runtime", action="store_true")
    ap.add_argument("--real-world", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--report-only")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    if a.report_only:
        print("report:", build_report(a.report_only))
        return 0
    if not a.config:
        ap.error("--config is required")
    cfg = deep_update(load_yaml(a.config), parse_overrides(a.overrides))
    if not a.skip_benchmark:
        res = run_benchmark(cfg)
        print("report:", build_report(res["output_dir"]))
    if a.runtime:
        run_runtime(cfg)
    if a.real_world:
        run_real_world(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
