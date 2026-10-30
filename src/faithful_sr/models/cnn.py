r"""Trainable super-resolution CNNs (implemented from scratch in PyTorch).

SRCNN  (Dong et al., ECCV 2014 / TPAMI 2016)
    Classic 3-layer network applied to a *bicubic-upsampled* input:
    patch extraction (9x9, 64) -> non-linear mapping (5x5, 32) -> reconstruction (5x5, 3).
    We use the 9-5-5 variant, 'same' padding and RGB input (the paper used Y only);
    the bicubic pre-upsampling uses PyTorch's bicubic (Keys a = -0.75).

FSRNet ("Faithful SR Net", this project)
    A compact residual network in the spirit of ESPCN (Shi et al., 2016) and
    EDSR (Lim et al., 2017):
      * all convolutions run at LR resolution (cheap: s^2 fewer pixels than SRCNN),
      * ``n_blocks`` residual blocks  x -> x + conv(ReLU(conv(x)))  without batch norm
        (EDSR showed BN hurts SR by normalising away contrast information),
      * sub-pixel upsampling: conv to 3*s^2 channels + PixelShuffle(s),
      * optional global bicubic skip: the network predicts the *residual*
        ``HR - bicubic(LR)``, i.e. only the missing high frequencies.
    It is *not* an exact reproduction of any published model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, scale: int = 3, f1: int = 9, f2: int = 5, f3: int = 5, n1: int = 64, n2: int = 32):
        super().__init__()
        self.scale = int(scale)
        self.conv1 = nn.Conv2d(3, n1, f1, padding=f1 // 2, padding_mode="replicate")
        self.conv2 = nn.Conv2d(n1, n2, f2, padding=f2 // 2, padding_mode="replicate")
        self.conv3 = nn.Conv2d(n2, 3, f3, padding=f3 // 2, padding_mode="replicate")
        self.kernel_sizes = (f1, f2, f3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        y = F.relu(self.conv1(up))
        y = F.relu(self.conv2(y))
        return up + self.conv3(y)   # residual on the bicubic estimate (stabilises training)

    def receptive_field_lr(self) -> int:
        """Receptive field in LR pixels (HR receptive field / scale, rounded up, + bicubic support)."""
        hr = sum(k - 1 for k in self.kernel_sizes) + 1
        return -(-hr // self.scale) + 4


class ResBlock(nn.Module):
    def __init__(self, c: int, res_scale: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.conv2(F.relu(self.conv1(x), inplace=True))


class FSRNet(nn.Module):
    def __init__(self, scale: int = 3, channels: int = 48, n_blocks: int = 8,
                 bicubic_skip: bool = True, res_scale: float = 1.0):
        super().__init__()
        self.scale, self.channels, self.n_blocks, self.bicubic_skip = int(scale), channels, n_blocks, bicubic_skip
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(channels, res_scale) for _ in range(n_blocks)])
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = nn.Sequential(nn.Conv2d(channels, 3 * self.scale ** 2, 3, padding=1),
                                      nn.PixelShuffle(self.scale))
        # start as "bicubic + small residual": zero the last conv so training starts at bicubic
        if bicubic_skip:
            nn.init.zeros_(self.upsample[0].weight)
            nn.init.zeros_(self.upsample[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.head(x)
        f = f + self.body_tail(self.body(f))         # long skip connection
        out = self.upsample(f)
        if self.bicubic_skip:
            out = out + F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return out

    def receptive_field_lr(self) -> int:
        n_convs = 1 + 2 * self.n_blocks + 1 + 1     # head + blocks + body_tail + upsample conv
        rf = 2 * n_convs + 1
        return rf + (4 if self.bicubic_skip else 0)


MODELS = {"srcnn": SRCNN, "fsrnet": FSRNet}


def build_model(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    name = cfg.pop("name", "fsrnet").lower()
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name](**cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
