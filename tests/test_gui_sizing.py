import json

import numpy as np
import pytest
from PIL import Image

from faithful_sr.inference.sizing import fit_canvas


def test_fit_canvas_pad_and_crop_never_stretch():
    img = np.random.default_rng(0).random((300, 300, 3)).astype(np.float32)  # square
    pad, info = fit_canvas(img, 384, 216, "pad")
    assert pad.shape == (216, 384, 3) and info["content_size"] == [216, 216]
    crop, info = fit_canvas(img, 384, 216, "crop")
    assert crop.shape == (216, 384, 3) and info["content_size"] == [384, 384]
    same, info = fit_canvas(img, 300, 300)
    assert info["canvas_mode"] == "exact"
    with pytest.raises(ValueError):
        fit_canvas(img, 10, 20, "stretch")


def test_gui_run_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from faithful_sr.gui import app
    monkeypatch.setattr(app, "OUT_DIR", tmp_path / "gui")
    p = tmp_path / "in.png"
    Image.fromarray((np.random.default_rng(0).random((40, 64, 3)) * 255).astype(np.uint8)).save(p)
    (ref, out), summary, meta, dst = app.run(str(p), "Bicubic", "3×", None, "Keep aspect ratio (input × scale)",
                                             "Pad (letterbox, keep everything)", 0, 24, "cnn_then_bicubic", "PNG", 95)
    assert out.shape == (120, 192, 3) and ref.shape == out.shape
    assert Image.open(dst).size == (192, 120)
    assert json.loads(meta)["output_size"] == [192, 120]
    (_, out4k), _, meta, _ = app.run(str(p), "Bicubic", "6×", None, "4K UHD (3840x2160)",
                                     "Pad (letterbox, keep everything)", 0, 24, "cnn_then_bicubic", "PNG", 95)
    assert json.loads(meta)["output_size"] == [3840, 2160]
    with pytest.raises(ValueError):
        app.run(str(p), "Trained CNN (checkpoint)", "3×", None, "Keep aspect ratio (input × scale)",
                "Pad", 0, 24, "cnn_then_bicubic", "PNG", 95)
    assert "Input" in app.describe(str(p))
    app.build_ui()  # UI constructs without launching
