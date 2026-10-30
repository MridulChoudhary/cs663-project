"""Command-line upscaling (same API the GUI uses).

Examples
--------
python scripts/upscale.py input.png --method bicubic --scale 3
python scripts/upscale.py input.png --method cnn --checkpoint outputs/runs/fsrnet_l1/best.pt --scale 6
python scripts/upscale.py folder_of_images/ --method bicubic --scale 3 --out outputs/upscaled
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faithful_sr.inference.upscale import available_methods, upscale_file  # noqa: E402
from faithful_sr.utils.image_io import SUPPORTED_EXTENSIONS, ImageLoadError  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--method", default="bicubic", help=f"one of {available_methods()}")
    ap.add_argument("--scale", type=int, default=3, choices=[2, 3, 4, 6])
    ap.add_argument("--out", default="outputs/upscaled", help="output folder")
    ap.add_argument("--checkpoint", default=None, help="checkpoint for learned methods")
    ap.add_argument("--tile", type=int, default=0, help="tile size in LR pixels (0 = whole image)")
    ap.add_argument("--overlap", type=int, default=16, help="tile overlap in LR pixels")
    ap.add_argument("--format", default="png", choices=["png", "jpg"])
    args = ap.parse_args(argv)

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS) if src.is_dir() else [src]
    kwargs = {}
    if args.checkpoint:
        kwargs["checkpoint"] = args.checkpoint
    if args.tile:
        kwargs.update(tile=args.tile, overlap=args.overlap)
    out_dir = Path(args.out)
    status = 0
    for f in files:
        dst = out_dir / f"{f.stem}_{args.method}_x{args.scale}.{args.format}"
        try:
            res = upscale_file(f, args.method, args.scale, dst, **kwargs)
        except (ImageLoadError, ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"[error] {f.name}: {e}", file=sys.stderr)
            status = 1
            continue
        dst.with_suffix(".json").write_text(json.dumps(res.metadata(), indent=2))
        print(f"{f.name}: {res.input_size} -> {res.output_size} via {res.method} "
              f"in {res.runtime_s:.2f}s -> {dst}")
        for w in res.warnings:
            print(f"  warning: {w}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
