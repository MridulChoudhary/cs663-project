"""End-to-end: prepare -> benchmark -> report on a tiny synthetic dataset (classical methods only)."""
import json

import numpy as np
import pandas as pd
from PIL import Image


def test_prepare_benchmark_report(tmp_path):
    from faithful_sr.data.prepare import prepare
    from faithful_sr.evaluation.benchmark import run_benchmark
    from faithful_sr.evaluation.report import build_report
    rng = np.random.default_rng(0)
    for split in ("tr", "te"):
        d = tmp_path / "data" / split
        d.mkdir(parents=True)
        for i in range(2):
            y, x = np.mgrid[0:132, 0:180]
            img = np.stack([np.sin(x / (3 + i)) * 0.5 + 0.5, (y % 17) / 17, rng.random((132, 180))], 2)
            Image.fromarray((img * 255).astype(np.uint8)).save(d / f"img {i}.png")
    (tmp_path / "data" / "te" / "tags.csv").write_text("filename,categories\nimg 0.png,poster;text_logo\nimg 1.png,texture\n")
    cfg = {"processed_dir": "proc", "scales": [3], "eval_degradations": ["bicubic"], "require_splits": ["test"],
           "subimage": {"size": 96, "step": 96},
           "sources": [{"name": "t", "dir": "data/tr", "split": "train"},
                       {"name": "poster", "dir": "data/te", "split": "test", "tags_file": "data/te/tags.csv"}]}
    prepare(cfg, tmp_path)
    rois = {"poster_img_0": [{"label": "text_logo", "box": [12, 12, 60, 48]}]}
    (tmp_path / "rois.json").write_text(json.dumps(rois))
    res = run_benchmark({"processed_dir": str(tmp_path / "proc"), "output_dir": str(tmp_path / "eval"),
                         "scales": [3], "degradations": ["bicubic"], "rois": str(tmp_path / "rois.json"),
                         "auto_face_exclude_prefixes": ["poster_"], "n_visual": 1,
                         "methods": [{"label": "Nearest", "method": "nearest"}, {"label": "Bicubic", "method": "bicubic"},
                                     {"label": "Missing", "method": "cnn", "kwargs": {"checkpoint": "nope.pt"}}]},
                        log=lambda m: None)
    m = pd.read_csv(tmp_path / "eval" / "metrics_per_image.csv")
    assert set(m.method) == {"Nearest", "Bicubic"} and len(m) == 4
    st = pd.read_csv(tmp_path / "eval" / "status.csv")
    assert (st[st.method == "Missing"].status == "not run").all()
    roi = pd.read_csv(tmp_path / "eval" / "roi_metrics.csv")
    assert "text_logo" in set(roi.roi) and "face" not in set(roi.roi)
    rep = build_report(res["output_dir"])
    assert (rep / "results.md").exists() and (rep / "fig_psnr_by_method.png").exists()
    assert "not run" in (rep / "results.md").read_text()


def test_benchmark_resume(tmp_path):
    from faithful_sr.data.prepare import prepare
    from faithful_sr.evaluation.benchmark import run_benchmark
    d = tmp_path / "data" / "te"
    d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(3):
        Image.fromarray((rng.random((120, 150, 3)) * 255).astype(np.uint8)).save(d / f"{i}.png")
    prepare({"processed_dir": "proc", "scales": [3], "eval_degradations": ["bicubic"], "require_splits": ["test"],
             "sources": [{"name": "p", "dir": "data/te", "split": "test"}]}, tmp_path)
    cfg = {"processed_dir": str(tmp_path / "proc"), "output_dir": str(tmp_path / "eval"), "scales": [3],
           "degradations": ["bicubic"], "n_visual": 1, "methods": [{"label": "Nearest", "method": "nearest"}]}
    q = lambda m: None  # noqa: E731
    run_benchmark(cfg, log=q)
    mp = tmp_path / "eval" / "metrics_per_image.csv"
    assert len(pd.read_csv(mp)) == 3
    # simulate an interruption: drop the last result line, then add a second method and rerun
    lines = mp.read_text().splitlines()
    mp.write_text("\n".join(lines[:-1]) + "\n")
    cfg["methods"].append({"label": "Bicubic", "method": "bicubic"})
    logs = []
    run_benchmark(cfg, log=logs.append)
    m = pd.read_csv(mp)
    assert len(m) == 6 and not m.duplicated(["image_id", "method"]).any()
    assert any("resuming: 2" in l for l in logs)
    # nothing left to do: rerun adds no rows
    run_benchmark(cfg, log=q)
    assert len(pd.read_csv(mp)) == 6
    roi = pd.read_csv(tmp_path / "eval" / "roi_metrics.csv")
    assert not roi.duplicated(["image_id", "method", "roi"]).any()
    # resume: false starts over
    run_benchmark({**cfg, "resume": False}, log=q)
    assert len(pd.read_csv(mp)) == 6
