# 09 — Decisions Log and Working Environment

This file records decisions made after the original prompt pack (mirrored in the claude.ai project as `claude/09_DECISIONS_AND_ENVIRONMENT.md`). It overrides earlier prompt files where they conflict. Update it whenever a new expectation, goal or change is agreed, *before* implementing it.

## Grading target (CS663 "What is Expected from a Course Project")
Every stage must feed one of the six graded items, and the 10-minute presentation must show all six:
1. Read papers — SRCNN (Dong et al., 2014/2016), ESPCN (sub-pixel upsampling, Shi et al., 2016), EDSR (Lim et al., 2017), Real-ESRGAN (Wang et al., 2021), Blau & Michaeli "Perception–Distortion Tradeoff" (2018).
2. Understand the maths — interpolation kernels as convolution with a reconstruction kernel; SR as ill-posed inverse problem y = Q((x∗k)↓s + n); L1 vs L2 as Laplacian vs Gaussian MLE (median vs mean); PSNR/SSIM; perception–distortion trade-off. See `docs/methodology.md`.
3. Implement from scratch — NumPy interpolation (library only as a test oracle), our own CNNs + training loop, degradation pipeline, metrics cross-checked against scikit-image, tiled inference.
4. Datasets — DIV2K + user-supplied HR posters/artwork (categorised test benchmark) + real-world LR folder (qualitative only).
5. Comprehensive empirical analysis + ablations — see "Experiment scope".
6. Summarise and explain results — report + slides built from saved outputs only.

## Agreed decisions
| Topic | Decision |
|---|---|
| Training data | DIV2K train HR (800) — downloaded inside Colab (cached on Drive), never committed. |
| Validation / test (photos) | DIV2K valid (100) split 50/50 by seeded source-level split into **val** (checkpoint selection) and **test** (natural-photo benchmark), so results do not depend only on how many posters are supplied. *(added 2026-10-31 during build)* |
| Benchmark data | User supplies HR posters/artwork (~20–40) in `data/raw/posters/` + `tags.csv` with the six categories. Test split only. Not committed. |
| Real-world set | Optional `data/real_world/`; qualitative grids + observation template only. |
| Code home | Local folder `Desktop\cs663-project` (day-to-day) **and** GitHub `MridulChoudhary/cs663-project` (pushed at stage milestones). Colab clones GitHub. |
| Compute | CPU smoke tests in Claude's workspace; real training/eval on Colab free GPU, outputs+models symlinked to Drive, `--resume auto`. |
| Deadline | Within 1 week → E1–E8 + A1 (L1 vs MSE) + A2 (bicubic vs mixed training). SRCNN baseline if time. Depth ablation / direct-6× model out of scope unless time remains. |
| GUI | Gradio 6 (`gr.ImageSlider`), thin layer over `inference/upscale.py`. |
| CNN | FSRNet (ours): 48 ch, 8 residual blocks, PixelShuffle ×3, global bicubic skip (~0.37 M params). SRCNN 9-5-5 baseline. |
| Pretrained comparator | Real-ESRGAN x4plus (BSD-3) via `spandrel`, explicit download (`scripts/download_weights.py`). 3× = 4× then anti-aliased bicubic ↓ to 3×; 6× = 4× then bicubic ↑1.5. |
| 6× strategy | Bicubic 6×; CNN 3× → bicubic 2× (default); CNN 3× → CNN 3× → anti-aliased ↓ to 6×; Real-ESRGAN adapted. Seam-free tiling (overlap ≥ RF/2). |
| Metrics | PSNR/SSIM on Y (BT.601), crop = scale; RGB-PSNR; edge-PSNR; overshoot (ringing); LPIPS-alex (works on Colab; blocked in Claude's sandbox). Paired bootstrap CI + win rate vs bicubic. |
| ROIs | Manual `data/rois.json` for text/logos; automatic face (Haar cascade — OpenCV 5 wheels no longer ship it, fetched by `download_weights.py`), edges, texture, flat; boxes snapped to multiples of 6. |

## Experiment scope (1-week plan)
Required: E1 interpolation comparison; E2 bicubic vs CNN; E3 CNN vs Real-ESRGAN; E4 3× vs 6× strategies; E5 degradation robustness (bicubic and mixed LR test sets); E6 category-wise; E7 ROI; E8 runtime/memory. Ablations: A1 L1 vs MSE; A2 bicubic- vs mixed-trained. Status in `docs/experiments.md`.

## Workflow rule
When a new goal/expectation/change is introduced: (1) edit the relevant prompt file(s) or this log, (2) mirror into repo `docs/`, (3) implement, (4) test, (5) commit; push to GitHub at stage milestones.
