import copy

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from faithful_sr.evaluation import metrics as M  # noqa: E402
from faithful_sr.models.cnn import FSRNet, SRCNN, build_model, count_parameters  # noqa: E402


@pytest.mark.parametrize("scale", [3, 6])
@pytest.mark.parametrize("name", ["fsrnet", "srcnn"])
def test_model_output_shape(name, scale):
    m = build_model({"name": name, "scale": scale})
    y = m(torch.rand(2, 3, 13, 17))
    assert tuple(y.shape) == (2, 3, 13 * scale, 17 * scale)


def test_fsrnet_starts_as_bicubic():
    m = FSRNet(3, 16, 2, bicubic_skip=True)
    x = torch.rand(1, 3, 10, 10)
    up = torch.nn.functional.interpolate(x, scale_factor=3, mode="bicubic", align_corners=False)
    assert torch.allclose(m(x), up)


def test_param_count_and_rf():
    m = FSRNet(3, 48, 8)
    assert 300_000 < count_parameters(m) < 500_000
    assert m.receptive_field_lr() == 2 * (1 + 16 + 1 + 1) + 1 + 4
    assert SRCNN(3).receptive_field_lr() > 0


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model({"name": "vdsr"})


# ------------------------------------------------------------------ metrics
def test_psnr_ssim_match_skimage():
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    rng = np.random.default_rng(0)
    gt = rng.random((64, 80, 3)).astype(np.float32)
    pred = np.clip(gt + rng.normal(0, 0.05, gt.shape), 0, 1).astype(np.float32)
    yp, yg = M.rgb_to_y(pred), M.rgb_to_y(gt)
    assert M.psnr(pred, gt) == pytest.approx(peak_signal_noise_ratio(yg, yp, data_range=1.0), abs=1e-6)
    ref = structural_similarity(yg, yp, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                use_sample_covariance=False)
    assert M.ssim(pred, gt) == pytest.approx(ref, abs=1e-6)
    assert M.ssim(gt, gt) == pytest.approx(1.0)
    assert M.psnr(gt, gt) == float("inf")


def test_y_channel_range_and_crop():
    white = np.ones((10, 10, 3))
    assert M.rgb_to_y(white)[0, 0] == pytest.approx(235 / 255)
    with pytest.raises(ValueError):
        M.psnr(np.zeros((4, 4, 3)), np.zeros((5, 4, 3)))
    a = np.zeros((10, 10, 3))
    b = a.copy()
    b[0, 0] = 1  # only a border pixel differs
    assert M.psnr(a, b, crop_border=2) == float("inf")


# ----------------------------------------------------------------- training
def _make_data(tmp):
    rng = np.random.default_rng(0)
    from faithful_sr.data.degradation import degrade
    from faithful_sr.utils.image_io import save_image
    (tmp / "train").mkdir()
    for i in range(6):
        Image.fromarray((rng.random((60, 60, 3)) * 255).astype(np.uint8)).save(tmp / "train" / f"{i}.png")
    for i in range(2):
        hr = rng.random((48, 48, 3)).astype(np.float32)
        save_image(hr, tmp / "hr" / f"v{i}.png")
        save_image(degrade(hr, 3)[0], tmp / "lr" / f"v{i}.png")


def _cfg(tmp, out, **train_kw):
    t = {"device": "cpu", "seed": 1, "iterations": 8, "batch_size": 2, "lr": 1e-3, "loss": "l1",
         "val_every": 4, "ckpt_every": 2, "log_every": 2}
    t.update(train_kw)
    return {"output_dir": str(tmp / out),
            "model": {"name": "fsrnet", "scale": 3, "channels": 8, "n_blocks": 1},
            "data": {"train_dir": str(tmp / "train"), "val_lr_dir": str(tmp / "lr"), "val_hr_dir": str(tmp / "hr"),
                     "lr_patch": 12, "workers": 0, "degradation": "mixed"},
            "train": t}


def test_train_checkpoint_and_exact_resume(tmp_path):
    from faithful_sr.training.train import train
    _make_data(tmp_path)
    quiet = lambda m: None  # noqa: E731
    full = train(_cfg(tmp_path, "full"), log=quiet)
    assert full["iterations"] == 8
    for f in ("best.pt", "latest.pt", "ckpt_0000002.pt", "train_log.csv", "val_log.csv", "curves.png"):
        assert (tmp_path / "full" / f).exists()

    cfg = _cfg(tmp_path, "resumed", stop_after=5)   # interrupted mid-epoch (3 steps/epoch)
    r = train(cfg, log=quiet)
    assert r.get("interrupted") and r["iterations"] == 5
    cfg2 = copy.deepcopy(cfg)
    cfg2["train"].pop("stop_after")
    # resume from the iteration-4 checkpoint (latest written at it 4; it 5 not checkpointed)
    r2 = train(cfg2, resume="auto", log=quiet)
    assert r2["iterations"] == 8
    a = torch.load(tmp_path / "full" / "latest.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=False)["model"]
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_missing_checkpoint_message(tmp_path):
    from faithful_sr.training.checkpointing import load_checkpoint
    with pytest.raises(FileNotFoundError, match="Train a model first"):
        load_checkpoint(tmp_path / "nope.pt")


def test_restore_rng_accepts_non_cpu_style_tensors():
    """Checkpoints loaded with map_location='cuda' give GPU tensors; restore_rng must coerce to CPU uint8."""
    import random
    from faithful_sr.training.checkpointing import restore_rng
    torch.manual_seed(5)
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state().to(torch.int64), "cuda": None}   # wrong dtype, like a moved tensor
    a = torch.rand(3)
    restore_rng({"rng": state})
    assert torch.equal(torch.rand(3), a)
