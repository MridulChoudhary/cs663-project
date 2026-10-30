"""Launch the Gradio GUI.   python scripts/launch_gui.py [--share] [--port 7860]"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # so relative paths (models/, outputs/) resolve consistently

from faithful_sr.gui.app import main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public Gradio link (useful on Colab)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    main(a.share, a.port)
