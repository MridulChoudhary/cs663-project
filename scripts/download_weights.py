"""Explicitly download optional third-party files (never done automatically at startup).

* RealESRGAN_x4plus.pth (~64 MB) - Real-ESRGAN, Wang et al. 2021, BSD-3-Clause
  https://github.com/xinntao/Real-ESRGAN
* haarcascade_frontalface_default.xml - OpenCV (Apache-2.0), used only to place
  automatic 'face' ROIs (OpenCV 5 wheels no longer bundle it)

python scripts/download_weights.py [--only realesrgan|haar]
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "realesrgan": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                   ROOT / "models/pretrained/RealESRGAN_x4plus.pth"),
    "haar": ("https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
             ROOT / "models/pretrained/haarcascade_frontalface_default.xml"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(FILES))
    a = ap.parse_args(argv)
    for name, (url, dst) in FILES.items():
        if a.only and name != a.only:
            continue
        if dst.exists():
            print(f"{name}: already present at {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"{name}: downloading {url}")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
        print(f"  saved {dst} ({dst.stat().st_size / 2**20:.1f} MB, sha256 "
              f"{hashlib.sha256(dst.read_bytes()).hexdigest()[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
