r"""Training loop for the SR CNNs.

Loss functions
--------------
* L1  : mean |f(y) - x|   (MLE under Laplacian noise; robust, sharper edges in practice)
* MSE : mean (f(y) - x)^2 (MLE under Gaussian noise; directly maximises PSNR but
        averages plausible solutions -> blur)

Design
------
* Iteration-based schedule (``train.iterations``) with cosine LR decay.
* Validation every ``train.val_every`` iterations on fixed val pairs (Y-PSNR,
  crop = scale); the best-PSNR model is saved as ``best.pt``.
* ``latest.pt`` is overwritten atomically every ``ckpt_every`` iterations and a
  numbered copy ``ckpt_<iter>.pt`` is kept (nothing is deleted automatically).
* Exact mid-epoch resume: the sampler order is a function of (seed, epoch) and
  the patch randomness a function of (seed, epoch, index), so resuming at
  iteration k continues with exactly the batches an uninterrupted run would see.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from faithful_sr.data.patches import PairedImageDataset, TrainPatchDataset
from faithful_sr.evaluation.metrics import psnr
from faithful_sr.models.cnn import build_model, count_parameters
from faithful_sr.training.checkpointing import (find_resume, load_checkpoint, restore_rng,
                                                save_checkpoint)
from faithful_sr.utils.config import environment_info, set_seed, write_json
from faithful_sr.utils.image_io import float_to_tensor, tensor_to_float

LOSSES = {"l1": F.l1_loss, "mse": F.mse_loss}


class ResumableSampler(Sampler):
    """Seeded per-epoch permutation that can start part-way through an epoch."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self) -> int:
        return self.n - self.start


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available - falling back to CPU")
        return torch.device("cpu")
    return torch.device(pref)


@torch.no_grad()
def validate(model, ds: PairedImageDataset, device, max_images: int = 0) -> dict:
    model.eval()
    vals = []
    n = len(ds) if not max_images else min(max_images, len(ds))
    for i in range(n):
        _, lr, hr = ds.get_numpy(i)
        sr = tensor_to_float(model(float_to_tensor(lr).to(device)))
        vals.append(psnr(sr, hr, crop_border=ds.scale))
    model.train()
    return {"val_psnr_y": float(np.mean(vals)), "n_val": n}


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def train(cfg: dict, resume: Optional[str] = None, log=print) -> dict:
    tcfg, dcfg, mcfg = cfg["train"], cfg["data"], cfg["model"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(tcfg.get("seed", 663))
    set_seed(seed)
    device = pick_device(tcfg.get("device", "auto"))
    scale = int(mcfg["scale"])

    model = build_model(mcfg).to(device)
    train_ds = TrainPatchDataset(dcfg["train_dir"], scale, dcfg.get("lr_patch", 48), dcfg.get("degradation", "bicubic"),
                                 dcfg.get("augment", "flips_rot"), seed, dcfg.get("repeat", 1), dcfg.get("cache", False))
    val_ds = PairedImageDataset(dcfg["val_lr_dir"], dcfg["val_hr_dir"], scale, dcfg.get("val_center_crop", 0))
    sampler = ResumableSampler(len(train_ds), seed)
    bs = int(tcfg.get("batch_size", 16))
    workers = int(dcfg.get("workers", 2))

    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 2e-4)), betas=(0.9, 0.99))
    total = int(tcfg["iterations"])
    warmup = int(tcfg.get("warmup", 0))
    min_ratio = float(tcfg.get("min_lr_ratio", 0.01))

    def lr_lambda(it):
        if it < warmup:
            return (it + 1) / warmup
        p = (it - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = LOSSES[tcfg.get("loss", "l1").lower()]

    it, best = 0, -1.0
    ckpt_path = Path(resume) if resume and resume != "auto" else (find_resume(out) if resume == "auto" else None)
    if ckpt_path:
        st = load_checkpoint(ckpt_path, map_location=device)
        if st.get("model_config") != mcfg:
            raise RuntimeError(f"Checkpoint model config {st.get('model_config')} != current {mcfg}")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        if "scaler" in st:
            scaler.load_state_dict(st["scaler"])
        it, best = int(st["iteration"]), float(st["best_metric"])
        restore_rng(st)
        log(f"[resume] from {ckpt_path} at iteration {it} (best val PSNR {best:.3f})")
    else:
        write_json({"config": cfg, "environment": environment_info(),
                    "parameters": count_parameters(model),
                    "receptive_field_lr": model.receptive_field_lr()}, out / "run_info.json")
        log(f"[train] {mcfg} | params {count_parameters(model):,} | device {device} | "
            f"train patches/epoch {len(train_ds)} | val images {len(val_ds)}")

    steps_per_epoch = max(1, len(train_ds) // bs)
    val_every, ckpt_every, log_every = (int(tcfg.get(k, d)) for k, d in
                                        (("val_every", 1000), ("ckpt_every", 1000), ("log_every", 100)))
    model.train()
    t0, run_loss, run_n = time.time(), 0.0, 0
    while it < total:
        epoch, start_batch = divmod(it, steps_per_epoch)
        train_ds.set_epoch(epoch)
        sampler.set_epoch(epoch, start_batch * bs)
        loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=workers, drop_last=True,
                            pin_memory=device.type == "cuda", persistent_workers=False)
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(device, non_blocking=True), hr_b.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp, dtype=torch.float16):
                sr = model(lr_b)
                loss = loss_fn(sr.float(), hr_b)
            if not torch.isfinite(loss):
                save_checkpoint(out / "nan_debug.pt", model, iteration=it, epoch=epoch, best_metric=best, config=cfg)
                raise FloatingPointError(f"Non-finite loss at iteration {it}; lower train.lr or disable amp")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            run_loss += float(loss.item())
            run_n += 1

            if it % log_every == 0 or it == total:
                row = {"iteration": it, "epoch": epoch, "loss": run_loss / run_n,
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                _append_csv(out / "train_log.csv", row)
                log(f"[it {it}/{total}] loss {row['loss']:.5f} lr {row['lr']:.2e} {row['elapsed_s']}s")
                run_loss, run_n = 0.0, 0
            if it % val_every == 0 or it == total:
                v = validate(model, val_ds, device, int(dcfg.get("val_max_images", 0)))
                _append_csv(out / "val_log.csv", {"iteration": it, **v})
                improved = v["val_psnr_y"] > best
                if improved:
                    best = v["val_psnr_y"]
                    save_checkpoint(out / "best.pt", model, iteration=it, epoch=epoch, best_metric=best, config=cfg)
                log(f"[val it {it}] PSNR-Y {v['val_psnr_y']:.3f} dB {'(best)' if improved else ''}")
            if it % ckpt_every == 0 or it == total:
                for p in (out / "latest.pt", out / f"ckpt_{it:07d}.pt"):
                    save_checkpoint(p, model, opt, sched, scaler, iteration=it, epoch=epoch,
                                    best_metric=best, config=cfg)
            if it >= total:
                break
            if tcfg.get("stop_after") and it >= int(tcfg["stop_after"]):
                log(f"[train] stop_after={tcfg['stop_after']} reached (simulated interruption)")
                return {"iterations": it, "best_val_psnr_y": best, "output_dir": str(out), "interrupted": True}
    plot_curves(out)
    return {"iterations": it, "best_val_psnr_y": best, "output_dir": str(out)}


def plot_curves(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    tl, vl = out / "train_log.csv", out / "val_log.csv"
    if not tl.exists():
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    t = pd.read_csv(tl)
    ax[0].plot(t["iteration"], t["loss"], color="#1f5fa8")
    ax[0].set(xlabel="iteration", ylabel="training loss", title="Training loss")
    ax[0].grid(alpha=0.3)
    if vl.exists():
        v = pd.read_csv(vl)
        ax[1].plot(v["iteration"], v["val_psnr_y"], marker="o", ms=3, color="#c0392b")
        ax[1].set(xlabel="iteration", ylabel="PSNR-Y (dB)", title="Validation PSNR (Y)")
        ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)
