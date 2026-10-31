"""Tests for the poster-specific pipeline features (large images, alpha, ids, source filter, real-world refs)."""
import numpy as np
import pytest
from PIL import Image

from faithful_sr.data.manifest import scan_source, slugify
from faithful_sr.utils.image_io import flatten_alpha, load_image


def test_slugify_and_unique_ids(tmp_path):
    assert slugify("DC Eat Peace, Motherf#ckers! ") == "dc_eat_peace_motherf_ckers"
    assert slugify("MCU SMATSV - Heroic") == "mcu_smatsv_heroic"
    for n in ("A b.png", "a_b.png"):
        Image.fromarray(np.zeros((120, 120, 3), np.uint8) + (1 if n[0] == "A" else 2)).save(tmp_path / n)
    ids = [r.image_id for r in scan_source("s", tmp_path, tmp_path, "test", log=lambda m: None)]
    assert len(set(ids)) == 2 and all(" " not in i for i in ids)


@pytest.mark.parametrize("fmt", ["jpg", "png"])
def test_load_with_max_long_side(tmp_path, fmt):
    y, x = np.mgrid[0:1500, 0:2400]
    arr = np.stack([x % 256, y % 256, (x + y) % 256], 2).astype(np.uint8)
    p = tmp_path / f"big.{fmt}"
    Image.fromarray(arr).save(p)
    li = load_image(p, max_long_side=600)
    assert li.rgb.shape == (375, 600, 3)
    assert any("Reduced" in w for w in li.warnings)
    assert load_image(p).rgb.shape == (1500, 2400, 3)          # untouched by default
    with pytest.raises(Exception):
        load_image(p, max_pixels=1000)


def test_flatten_alpha(tmp_path):
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[..., :3] = 200
    rgba[:, :5, 3] = 255      # left half opaque, right half transparent
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    li = load_image(tmp_path / "a.png")
    out = flatten_alpha(li, (0, 0, 0))
    assert np.allclose(out[:, :5], 200 / 255) and np.allclose(out[:, 5:], 0)


def test_include_sources_filter(tmp_path):
    pytest.importorskip("torch")
    from faithful_sr.training.train import train
    rng = np.random.default_rng(0)
    (tmp_path / "sub").mkdir()
    for name in ("div2k_a_0_0.png", "div2k_b_0_0.png", "poster_c_0_0.png"):
        Image.fromarray((rng.random((60, 60, 3)) * 255).astype(np.uint8)).save(tmp_path / "sub" / name)
    from faithful_sr.data.degradation import degrade
    from faithful_sr.utils.image_io import save_image
    hr = rng.random((48, 48, 3)).astype(np.float32)
    save_image(hr, tmp_path / "hr" / "v.png")
    save_image(degrade(hr, 3)[0], tmp_path / "lr" / "v.png")
    cfg = {"output_dir": str(tmp_path / "run"), "model": {"name": "fsrnet", "scale": 3, "channels": 8, "n_blocks": 1},
           "data": {"train_dir": str(tmp_path / "sub"), "val_lr_dir": str(tmp_path / "lr"), "val_hr_dir": str(tmp_path / "hr"),
                    "lr_patch": 12, "workers": 0, "include_sources": ["nothing"]},
           "train": {"device": "cpu", "iterations": 2, "batch_size": 1, "val_every": 2, "ckpt_every": 2, "log_every": 1}}
    with pytest.raises(RuntimeError, match="No training sub-images"):
        train(cfg, log=lambda m: None)
    cfg["data"]["include_sources"] = ["div2k"]
    assert train(cfg, log=lambda m: None)["iterations"] == 2


def test_real_world_reference_prepared(tmp_path):
    from faithful_sr.data.prepare import prepare
    up = tmp_path / "data" / "up"
    up.mkdir(parents=True)
    Image.fromarray((np.random.default_rng(0).random((400, 600, 3)) * 255).astype(np.uint8)).save(up / "Some File .png")
    tr = tmp_path / "data" / "tr"
    tr.mkdir()
    for i in range(2):
        Image.fromarray((np.random.default_rng(i + 1).random((120, 120, 3)) * 255).astype(np.uint8)).save(tr / f"{i}.png")
    cfg = {"processed_dir": "proc", "scales": [3], "subimage": {"size": 96, "step": 96}, "require_splits": [],
           "real_world_scale": 3,
           "sources": [{"name": "t", "dir": "data/tr", "split": "train"},
                       {"name": "upscaled", "dir": "data/up", "split": "real_world", "has_gt": False,
                        "lr_downscale": 4, "lr_max_long_side": 120, "reference_upscaled": True}]}
    prepare(cfg, tmp_path)
    lr = load_image(tmp_path / "proc/real_world/LR/upscaled_some_file.png").rgb
    ref = load_image(tmp_path / "proc/real_world/REF/upscaled_some_file.png").rgb
    assert lr.shape[:2] == (80, 120)           # /4 would be 100x150 -> capped to 120 long side
    assert ref.shape[:2] == (240, 360)         # reference at exactly 3x the LR size
