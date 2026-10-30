import numpy as np
import pytest
from PIL import Image

from faithful_sr.inference.upscale import upscale_file
from faithful_sr.utils import image_io as io


def test_roundtrip_png(tmp_path):
    arr = (np.random.default_rng(1).random((8, 12, 3)) * 255).astype(np.uint8)
    p = tmp_path / "a.png"
    Image.fromarray(arr).save(p)
    img = io.load_image(p)
    assert img.rgb.shape == (8, 12, 3) and img.rgb.dtype == np.float32
    assert np.array_equal(io.to_uint8(img.rgb), arr)
    q = io.save_image(img.rgb, tmp_path / "b.png")
    assert np.array_equal(np.asarray(Image.open(q)), arr)


def test_grayscale_and_alpha(tmp_path):
    Image.fromarray(np.full((5, 6), 100, np.uint8), "L").save(tmp_path / "g.png")
    g = io.load_image(tmp_path / "g.png")
    assert g.rgb.shape == (5, 6, 3) and g.warnings
    rgba = np.zeros((5, 6, 4), np.uint8)
    rgba[..., 3] = 128
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    a = io.load_image(tmp_path / "a.png")
    assert a.alpha is not None and a.alpha.shape == (5, 6)


def test_exif_orientation_applied(tmp_path):
    arr = np.zeros((4, 10, 3), np.uint8)
    pil = Image.fromarray(arr)
    exif = pil.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    pil.save(tmp_path / "r.jpg", exif=exif)
    assert io.load_image(tmp_path / "r.jpg").rgb.shape[:2] == (10, 4)


def test_errors(tmp_path):
    with pytest.raises(io.ImageLoadError):
        io.load_image(tmp_path / "missing.png")
    (tmp_path / "x.gif").write_bytes(b"GIF89a")
    with pytest.raises(io.ImageLoadError):
        io.load_image(tmp_path / "x.gif")
    (tmp_path / "bad.png").write_bytes(b"not a png")
    with pytest.raises(io.ImageLoadError):
        io.load_image(tmp_path / "bad.png")


def test_tensor_roundtrip():
    torch = pytest.importorskip("torch")
    img = np.random.default_rng(0).random((7, 9, 3)).astype(np.float32)
    t = io.float_to_tensor(img)
    assert tuple(t.shape) == (1, 3, 7, 9) and t.dtype == torch.float32
    assert np.allclose(io.tensor_to_float(t), img)


def test_upscale_file_does_not_overwrite(tmp_path):
    p = tmp_path / "in.png"
    Image.fromarray(np.zeros((4, 5, 3), np.uint8)).save(p)
    res = upscale_file(p, "bicubic", 3, tmp_path / "out.png")
    assert res.output_size == (15, 12)
    assert Image.open(tmp_path / "out.png").size == (15, 12)
    with pytest.raises(ValueError):
        upscale_file(p, "bicubic", 3, p)
    with pytest.raises(ValueError):
        upscale_file(p, "bicubic", 5)
