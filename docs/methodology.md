# Methodology

This document explains the mathematics behind every component and maps each to the code.
It is the source for the "Background" and "Methods" sections of the report and slides.

## 1. Super-resolution as an ill-posed inverse problem
We observe a low-resolution image **y** generated from an unknown high-resolution image **x**:

  y = Q( (x ∗ k) ↓ₛ + n )

k = blur kernel, ↓ₛ = downsampling by s per dimension, n = noise, Q = quantisation / JPEG.
The operator H = ↓ₛ∘(k∗·) maps an s²·N-pixel image to N pixels, so it has a huge null space: infinitely many x
explain the same y. Any SR method therefore chooses one solution using a *prior*:

* **Interpolation** uses a smoothness prior (x is band-limited / locally polynomial).
* **A CNN trained with L1/L2** learns E-like estimators of p(x | y) from data; with L2 it approximates the
  posterior **mean** (MMSE), which averages plausible textures → blur but high PSNR.
* **GAN-based models (Real-ESRGAN)** sample something close to the posterior **mode / a plausible sample**:
  sharp and realistic-looking, but details may be invented ("hallucinated").

Blau & Michaeli (CVPR 2018) proved that distortion (e.g. MSE) and perceptual quality (distance between the
distribution of outputs and natural images) cannot both be optimal: the **perception–distortion trade-off**.
This is the central lens of our analysis (`fig_perception_distortion.png`).

## 2. Classical interpolation (implemented from scratch — `models/interpolation.py`)
Resampling = convolution of the samples with a reconstruction kernel:
f̂(t) = Σⱼ f[j]·h(t − j), with output pixel i at input position c = (i + 0.5)/s − 0.5 (pixel-centre convention).

| Method | Kernel h(t) | Support | Property |
|---|---|---|---|
| Nearest | box: 1 for \|t\| < ½ | 1 | blocky, no new values |
| Bilinear | triangle: 1 − \|t\| | 2 | C⁰, blurs |
| Bicubic (Keys 1981, a = −0.5) | (a+2)\|t\|³ − (a+3)\|t\|² + 1 for \|t\|≤1; a\|t\|³ − 5a\|t\|² + 8a\|t\| − 4a for 1<\|t\|<2 | 4 | C¹, reproduces quadratics, negative lobes → slight **ringing/overshoot** at edges |
| Lanczos-3 | sinc(t)·sinc(t/3) | 6 | used only to resize Real-ESRGAN output |

Kernels are separable, so a resize is two sparse matrix products: **out = W_h · img · W_wᵀ**, where each row of
W holds the normalised kernel weights for one output pixel. When **downsampling**, the kernel is stretched by
1/s (anti-aliasing low-pass), matching MATLAB `imresize` — the standard way SR benchmarks create LR images.
Verified against Pillow to within Pillow's own 8-bit intermediate rounding (`tests/test_interpolation.py`).

## 3. Degradation pipeline (`data/degradation.py`)
Fixed, recorded order: Gaussian blur on HR (σ ∈ [0.4, 1.6]) → anti-aliased bicubic ↓3 or ↓6 → Gaussian noise
(σ ∈ [1, 6]/255) → 8-bit quantisation → JPEG (q ∈ [55, 95]). Presets: `bicubic` (clean benchmark),
`blur`, `noise`, `jpeg` (one factor each) and `mixed` (each op with probability 0.7/0.5/0.7). The blur is
symmetric and centred, so HR/LR alignment is exact (unit-tested by tracking a Gaussian blob's centroid).
Every random draw is a function of (seed, image id, variant) → bit-exact regeneration.

## 4. CNNs (implemented from scratch — `models/cnn.py`)
**SRCNN** (Dong et al. 2014/2016): bicubic upsample, then 9×9×64 → ReLU → 5×5×32 → ReLU → 5×5×3. Three layers
interpreted as patch extraction, non-linear mapping, reconstruction. Runs at HR resolution (expensive).
Differences from the paper: RGB instead of Y, residual connection to the bicubic input, Adam + L1.

**FSRNet** (ours, ESPCN/EDSR-inspired; not a reproduction):
* head 3×3 conv (3→C), **n residual blocks** (conv–ReLU–conv + identity, no batch-norm), body conv + long skip;
* **sub-pixel convolution**: conv C → 3s² channels, then PixelShuffle rearranges (3s², H, W) → (3, sH, sW).
  All convolutions run at LR resolution — s² = 9× fewer pixels than SRCNN;
* **global bicubic skip**: the net predicts the residual r = x − bicubic(y), i.e. only the missing high
  frequencies; the last conv is zero-initialised so training starts exactly at bicubic.
* Default C = 48, n = 8 → ≈ 0.37 M parameters; receptive field 2·19+1 = 39 LR px (+4 for the bicubic branch).

## 5. Losses (`training/train.py`)
* **L2 / MSE**: MLE under Gaussian noise; optimises PSNR directly; minimiser = conditional mean → averaging blur.
* **L1**: MLE under Laplacian noise; minimiser = conditional median → empirically sharper edges, more robust to
  outlier pixels. Ablation A1 compares the two with everything else fixed.

Training: Adam (β = 0.9, 0.99), cosine LR with warm-up, 30k iterations, batch 16, 48×48 LR patches,
flips/rot90 augmentation, gradient clipping 1.0, best checkpoint chosen on validation PSNR-Y.

## 6. Metrics (`evaluation/metrics.py`)
* **PSNR** = 10·log₁₀(1/MSE) on BT.601 luma Y, border crop = s.
* **SSIM** (Wang et al. 2004): l·c·s = [(2μₓμᵧ + C₁)(2σₓᵧ + C₂)] / [(μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂)],
  11×11 Gaussian window σ = 1.5, C₁ = (0.01)², C₂ = (0.03)²; matches scikit-image (unit-tested).
* **LPIPS** (Zhang et al. 2018): distance between deep AlexNet features, calibrated to human judgements.
* **Edge-PSNR**: PSNR on the 10 % strongest ground-truth Sobel-gradient pixels (text, lines).
* **Overshoot**: mean amount the output leaves the local 3×3 [min, max] range of the GT → ringing/halo indicator.
None of these proves that a detail is authentic; PSNR/SSIM measure agreement with *one* ground truth.

## 7. 6× strategies and tiling (`inference/learned.py`, `inference/tiling.py`)
6× = 3× CNN then bicubic 2× (default), or 3× CNN twice (9×) then anti-aliased ↓1.5. Both propagate first-stage
errors; the second also feeds the CNN its own output (distribution shift). Real-ESRGAN (native 4×): 3× = 4× then
anti-aliased resize to 3×; 6× = 4× then bicubic 1.5×.
Tiling processes cores with a halo of ≥ RF/2 LR pixels and keeps only the core → identical to whole-image
inference for our CNNs (max diff < 1e-5, `tests/test_tiling.py`).

## References
Keys 1981 (cubic convolution); Wang et al. 2004 (SSIM); Dong et al. 2014/2016 (SRCNN); Shi et al. 2016 (ESPCN);
Lim et al. 2017 (EDSR); Agustsson & Timofte 2017 (DIV2K); Blau & Michaeli 2018 (perception–distortion);
Zhang et al. 2018 (LPIPS); Wang et al. 2021 (Real-ESRGAN).
