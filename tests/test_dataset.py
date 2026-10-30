import numpy as np
import pytest
from PIL import Image

from faithful_sr.data import degradation as dg
from faithful_sr.data import manifest as mf
from faithful_sr.data.patches import extract_subimages


def _blob(h, w, cy, cx, s=4.0):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s * s))
    return np.repeat(g[..., None], 3, 2).astype(np.float32)


def _centroid(img):
    g = img[..., 0].astype(np.float64)
    y, x = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    return (g * y).sum() / g.sum(), (g * x).sum() / g.sum()


@pytest.mark.parametrize("scale", [3, 6])
@pytest.mark.parametrize("preset", ["bicubic", "blur"])
def test_lr_hr_alignment(scale, preset):
    """The LR blob centre, mapped back with the pixel-centre convention, matches HR."""
    hr = _blob(120, 132, 61.3, 70.8, s=6.0)
    lr, _ = dg.degrade(hr, scale, preset, np.random.default_rng(0))
    cy, cx = _centroid(lr)
    assert abs((cy + 0.5) * scale - 0.5 - 61.3) < 0.1
    assert abs((cx + 0.5) * scale - 0.5 - 70.8) < 0.1


def test_degradation_deterministic_and_recorded():
    hr = np.random.default_rng(0).random((60, 60, 3)).astype(np.float32)
    a, pa = dg.degrade(hr, 3, "mixed", dg.image_rng(1, "img", 0))
    b, pb = dg.degrade(hr, 3, "mixed", dg.image_rng(1, "img", 0))
    c, _ = dg.degrade(hr, 3, "mixed", dg.image_rng(1, "other", 0))
    assert np.array_equal(a, b) and pa == pb
    assert not np.array_equal(a, c)
    assert a.shape == (20, 20, 3)
    assert np.allclose(a * 255, np.round(a * 255), atol=1e-4)  # 8-bit quantised


def test_each_op_applied_when_prob_one():
    hr = np.random.default_rng(0).random((60, 60, 3)).astype(np.float32)
    for name, key in (("blur", "blur_sigma"), ("noise", "noise_sigma_255"), ("jpeg", "jpeg_quality")):
        _, p = dg.degrade(hr, 3, name, np.random.default_rng(1))
        assert key in p
    _, p = dg.degrade(hr, 3, "bicubic")
    assert not {"blur_sigma", "noise_sigma_255", "jpeg_quality"} & set(p)


def test_invalid_degradation_settings():
    with pytest.raises(ValueError):
        dg.DegradationConfig.from_dict({"blur_sigma": {"prob": 1.5, "low": 0.5, "high": 1}})
    with pytest.raises(ValueError):
        dg.DegradationConfig.from_dict({"jpeg_quality": {"prob": 1, "low": 90, "high": 50}})
    with pytest.raises(ValueError):
        dg.get_preset("does_not_exist")
    with pytest.raises(ValueError):
        dg.degrade(np.zeros((10, 10, 3), np.float32), 3)  # not divisible


def _write(dirpath, name, arr):
    dirpath.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dirpath / name)


def test_manifest_splits_duplicates_and_leakage(tmp_path):
    rng = np.random.default_rng(0)
    imgs = [(rng.random((120, 120, 3)) * 255).astype(np.uint8) for _ in range(10)]
    for i, im in enumerate(imgs):
        _write(tmp_path / "a", f"{i}.png", im)
    _write(tmp_path / "b", "dup.png", imgs[0])              # exact duplicate of a/0
    _write(tmp_path / "b", "tiny.png", imgs[1][:50, :50])   # below min size
    (tmp_path / "b" / "broken.png").write_bytes(b"xx")      # corrupt
    rows = mf.scan_source("a", tmp_path / "a", tmp_path, "auto", log=lambda m: None)
    mf.assign_auto_splits(rows, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    rows += mf.scan_source("b", tmp_path / "b", tmp_path, "test", log=lambda m: None)
    mf.mark_duplicates(rows, log=lambda m: None)
    mf.check_leakage(rows)
    splits = {r.image_id: r.split for r in rows}
    assert sum(s == "train" for s in splits.values()) == 6
    assert splits["b_tiny"] == "excluded" and splits["b_broken"] == "excluded"
    assert "excluded" in (splits["b_dup"], splits["a_0"])
    # determinism of the seeded split
    rows2 = mf.scan_source("a", tmp_path / "a", tmp_path, "auto", log=lambda m: None)
    mf.assign_auto_splits(rows2, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    assert [r.split for r in rows2] == [r.split for r in rows[:10]]
    # manifest round trip
    mf.write_manifest(rows, tmp_path / "m.csv")
    assert [r.split for r in mf.read_manifest(tmp_path / "m.csv")] == [r.split for r in rows]
    # leakage detection
    rows[0].split, rows[1].split = "train", "test"
    rows[1].sha1 = rows[0].sha1
    with pytest.raises(AssertionError):
        mf.check_leakage(rows)


def test_bad_category_tag(tmp_path):
    (tmp_path / "tags.csv").write_text("filename,categories\nx.png,poster;cartoon\n")
    with pytest.raises(ValueError):
        mf.read_tags(tmp_path / "tags.csv")


def test_subimages_cover_image():
    hr = np.zeros((500, 700, 3), np.float32)
    tiles = extract_subimages(hr, 240, 240)
    ys = {y for y, _, _ in tiles}
    xs = {x for _, x, _ in tiles}
    assert max(ys) + 240 == 500 and max(xs) + 240 == 700
    assert all(t.shape == (240, 240, 3) for _, _, t in tiles)


def test_train_patch_dataset_shapes_and_determinism(tmp_path):
    torch = pytest.importorskip("torch")
    from faithful_sr.data.patches import TrainPatchDataset
    rng = np.random.default_rng(0)
    for i in range(3):
        _write(tmp_path, f"{i}.png", (rng.random((100, 130, 3)) * 255).astype(np.uint8))
    ds = TrainPatchDataset(tmp_path, scale=3, lr_patch=16, degradation="mixed", repeat=2)
    assert len(ds) == 6
    lr, hr = ds[4]
    assert tuple(lr.shape) == (3, 16, 16) and tuple(hr.shape) == (3, 48, 48)
    assert lr.dtype == torch.float32
    assert torch.equal(ds[4][0], lr)
    ds.set_epoch(1)
    assert not torch.equal(ds[4][1], hr)


def test_empty_training_folder(tmp_path):
    pytest.importorskip("torch")
    from faithful_sr.data.patches import TrainPatchDataset
    with pytest.raises(RuntimeError):
        TrainPatchDataset(tmp_path, scale=3)
