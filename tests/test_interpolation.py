import numpy as np
import pytest
from PIL import Image

from faithful_sr.models import interpolation as interp

PIL_FILTERS = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
               "bicubic": Image.BICUBIC, "lanczos3": Image.LANCZOS}


def _rand(h=37, w=53, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


def _pillow_like(img_u8, oh, ow, method):
    """Our resize with Pillow's 8-bit rounding after the horizontal pass."""
    x = img_u8.astype(np.float32) / 255
    t = interp.resize(x, x.shape[0], ow, method, clip=False)
    t = (np.clip(np.floor(t * 255 + 0.5), 0, 255) / 255).astype(np.float32)
    return interp.resize(t, oh, ow, method) * 255


@pytest.mark.parametrize("method", list(PIL_FILTERS))
@pytest.mark.parametrize("size", [(111, 159), (74, 106), (12, 17), (222, 318)])
def test_matches_pillow(method, size):
    img = _rand()
    oh, ow = size
    ref = np.asarray(Image.fromarray(img).resize((ow, oh), PIL_FILTERS[method])).astype(np.float32)
    ours = _pillow_like(img, oh, ow, method)
    assert np.abs(ours - ref).max() <= 1.5  # within 1.5 grey levels of 255


@pytest.mark.parametrize("method", interp.METHODS)
@pytest.mark.parametrize("scale", [2, 3, 6])
def test_output_dimensions_and_range(method, scale):
    img = _rand(20, 31).astype(np.float32) / 255
    out = interp.upscale(img, scale, method)
    assert out.shape == (20 * scale, 31 * scale, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("method", interp.METHODS)
def test_constant_image_preserved(method):
    img = np.full((9, 13, 3), 0.3, np.float32)
    assert np.allclose(interp.upscale(img, 3, method), 0.3, atol=1e-5)


def test_deterministic_and_grayscale_2d():
    img = _rand(10, 10)[..., 0].astype(np.float32) / 255
    a = interp.upscale(img, 3, "bicubic")
    b = interp.upscale(img, 3, "bicubic")
    assert a.shape == (30, 30) and np.array_equal(a, b)


def test_nearest_replicates_pixels():
    img = np.arange(6, dtype=np.float32).reshape(2, 3) / 10
    out = interp.upscale(img, 3, "nearest")
    assert np.array_equal(out, np.kron(img, np.ones((3, 3), np.float32)))


def test_bilinear_linear_ramp_exact_in_interior():
    ramp = np.tile(np.linspace(0, 1, 20, dtype=np.float32), (5, 1))
    out = interp.upscale(ramp, 3, "bilinear")
    centres = (np.arange(60) + 0.5) / 3 - 0.5
    expected = np.interp(centres, np.arange(20), np.linspace(0, 1, 20))
    assert np.allclose(out[2, 3:-3], expected[3:-3], atol=1e-5)


def test_kernel_partition_of_unity():
    for m in ("bilinear", "bicubic", "lanczos3"):
        w = interp.weight_matrix(17, 51, m)
        assert np.allclose(np.asarray(w.sum(axis=1)).ravel(), 1.0)


def test_downscale_and_modcrop():
    img = _rand(31, 47).astype(np.float32) / 255
    with pytest.raises(ValueError):
        interp.downscale(img, 3)
    c = interp.modcrop(img, 3)
    assert c.shape[:2] == (30, 45)
    assert interp.downscale(c, 3).shape == (10, 15, 3)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        interp.resize(np.zeros((2, 2, 2, 2), np.float32), 4, 4)
    with pytest.raises(ValueError):
        interp.resize(np.zeros((4, 4), np.float32), 8, 8, "spline42")
    with pytest.raises(ValueError):
        interp.upscale(np.zeros((4, 4), np.float32), 0)
