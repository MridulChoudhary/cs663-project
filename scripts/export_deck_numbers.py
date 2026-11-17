"""Export the numbers used on the slides to outputs/report/deck_numbers.json.

Reads the benchmark CSVs through build_report's loaders, so the slides and the
report quote identical values. Run after build_report.py's inputs exist:
    python scripts/export_deck_numbers.py [--run outputs/eval/main]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/eval/main")
    ap.add_argument("--out", default="outputs/report/deck_numbers.json")
    a = ap.parse_args()
    m, r, rt = B.load(Path(a.run))
    P = lambda *x, **k: B.paired(m, *x, **k)  # noqa: E731
    M = lambda *x, **k: B.mean(m, *x, **k)  # noqa: E731
    d: dict = {}

    def put(k, p):
        d[k] = {kk: (int(v) if kk == "n" else float(v)) for kk, v in p.items()}

    put("e2", P("FSRNet-L1", "Bicubic", 3, "bicubic"))
    put("e2p", P("FSRNet-L1", "Bicubic", 3, "bicubic", "poster"))
    put("e2d", P("FSRNet-L1", "Bicubic", 3, "bicubic", "div2k"))
    put("e3p", P("FSRNet-L1", "Real-ESRGAN", 3, "bicubic", "poster"))
    put("e3d", P("FSRNet-L1", "Real-ESRGAN", 3, "bicubic", "div2k"))
    put("a1", P("FSRNet-L1", "FSRNet-MSE", 3, "bicubic"))
    put("a2m", P("FSRNet-L1-mixed", "FSRNet-L1", 3, "mixed"))
    put("a2c", P("FSRNet-L1-mixed", "FSRNet-L1", 3, "bicubic"))
    put("a3p", P("FSRNet-L1+posters", "FSRNet-L1", 3, "bicubic", "poster"))
    put("a3d", P("FSRNet-L1+posters", "FSRNet-L1", 3, "bicubic", "div2k"))
    put("arch", P("FSRNet-L1", "SRCNN", 3, "bicubic"))
    put("six", P("FSRNet-L1 3x>3x>down", "FSRNet-L1", 6, "bicubic"))

    meths = ["Nearest", "Bilinear", "Bicubic", "SRCNN", "FSRNet-L1", "Real-ESRGAN"]
    srcs = ["div2k", "poster"]
    d["bysrc"] = {s: {mm: M(mm, 3, "bicubic", src=s) for mm in meths} for s in srcs}
    d["lpips_src"] = {s: {mm: M(mm, 3, "bicubic", "lpips", src=s) for mm in meths} for s in srcs}
    a2m = ["Bicubic", "FSRNet-L1", "FSRNet-L1-mixed", "Real-ESRGAN"]
    d["a2bars"] = {g: {mm: M(mm, 3, g) for mm in a2m} for g in ["bicubic", "mixed"]}
    d["lpips_mixed"] = {mm: M(mm, 3, "mixed", "lpips") for mm in ["FSRNet-L1-mixed", "Real-ESRGAN"]}

    rp = r[(r.scale == 3) & (r.degradation == "bicubic") & r.image_id.str.startswith("poster")]
    labs = ["text_small", "text_logo", "face", "edges", "texture"]
    sel = lambda lab, mm: rp[(rp.roi == lab) & (rp.method == mm)]  # noqa: E731
    d["roi"] = {lab: {mm: float(sel(lab, mm).psnr_y.mean()) for mm in ["Bicubic", "SRCNN", "FSRNet-L1", "Real-ESRGAN"]} for lab in labs}
    d["roi_n"] = {lab: int(len(sel(lab, "Bicubic"))) for lab in labs}
    d["overshoot_ratio"] = {lab: float(sel(lab, "Real-ESRGAN").overshoot.mean() / sel(lab, "FSRNet-L1").overshoot.mean()) for lab in labs}

    rt7 = rt[rt.input == "1280x720"]
    one = lambda mm, s: rt7[(rt7.method == mm) & (rt7.scale == s)]  # noqa: E731
    d["runtime"] = {f"{mm}|{s}": float(one(mm, s).median_s.iloc[0])
                    for mm in ["Bicubic", "SRCNN", "FSRNet-L1", "Real-ESRGAN"] for s in (3, 6)}
    d["runtime"]["FSRNet-L1 3x>3x>down|6"] = float(rt7[rt7.method == "FSRNet-L1 3x>3x>down"].median_s.iloc[0])
    d["mem"] = {mm: float(one(mm, 3).peak_gpu_mb.iloc[0]) for mm in ["SRCNN", "FSRNet-L1", "Real-ESRGAN"]}
    d["x36"] = {mm: {s: M(mm, s, "bicubic") for s in (3, 6)} for mm in ["Bicubic", "FSRNet-L1", "Real-ESRGAN"]}
    d["n_results"] = int(len(m))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=1))
    print("[deck] wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
