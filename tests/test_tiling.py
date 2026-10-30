import numpy as np
import pytest

from faithful_sr.inference.tiling import tiled_upscale
from faithful_sr.models import interpolation as interp

torch = pytest.importorskip("torch")


def _patterns(h=70, w=95):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    flat = np.full((h, w, 3), 0.42, np.float32)
    grad = np.stack([x / w, y / h, (x + y) / (h + w)], 2).astype(np.float32)
    checker = (((x // 4 + y // 4) % 2)[..., None].repeat(3, 2) * 0.8 + 0.1).astype(np.float32)
    noise = np.random.default_rng(0).random((h, w, 3)).astype(np.float32)
    return {"flat": flat, "gradient": grad, "checker": checker, "noise": noise}


def _cnn_fn():
    from faithful_sr.models.cnn import FSRNet
    torch.manual_seed(0)
    m = FSRNet(3, 8, 2, bicubic_skip=True).eval()
    for p in m.parameters():                   # non-trivial weights
        torch.nn.init.normal_(p, 0, 0.05)
    from faithful_sr.utils.image_io import float_to_tensor, tensor_to_float

    def fn(img):
        with torch.no_grad():
            return tensor_to_float(m(float_to_tensor(img)))
    return m, fn


@pytest.mark.parametrize("name", ["flat", "gradient", "checker", "noise"])
@pytest.mark.parametrize("tile", [16, 23, 40])
def test_tiled_equals_whole_cnn(name, tile):
    m, fn = _cnn_fn()
    img = _patterns()[name]
    whole = fn(img)
    tiled = tiled_upscale(img, fn, 3, tile, overlap=m.receptive_field_lr() // 2 + 2)
    assert tiled.shape == whole.shape
    assert np.abs(tiled - whole).max() < 1e-5


def test_insufficient_overlap_creates_measurable_seams():
    m, fn = _cnn_fn()
    img = _patterns()["noise"]
    diff = np.abs(tiled_upscale(img, fn, 3, 16, overlap=0) - fn(img)).max()
    assert diff > 1e-3   # documents why the overlap matters


def test_tiled_classical_and_small_images():
    img = _patterns(30, 41)["gradient"]
    f = lambda t: interp.upscale(t, 3, "bicubic")  # noqa: E731
    assert np.abs(tiled_upscale(img, f, 3, 8, overlap=4) - f(img)).max() < 1e-5
    small = img[:5, :7]
    assert tiled_upscale(small, f, 3, 64, 4).shape == (15, 21, 3)
    with pytest.raises(ValueError):
        tiled_upscale(img, f, 3, 8, overlap=-1)
