# Experiments — plan, commands and status

**Final run: `outputs/eval/main` (Colab T4, 2026-11-16/17), 2,508 result rows. Report: `outputs/report/CS663_Faithful_SR_Report.pdf`; slides: `outputs/report/CS663_Faithful_SR_Presentation.pptx`.**

Status legend: ✅ done (results saved) · ⏳ pending (not yet run) · ⚙️ pipeline verified on synthetic sample only.
**No result below is a real number until its status is ✅** — values come only from `outputs/eval/main/report/`.

| ID | Question | How (config / command) | Status |
|---|---|---|---|
| E1 | Nearest vs bilinear vs bicubic | `configs/eval.yaml` methods 1–3 | ✅ bicubic best of the three (x3 clean: 30.6 dB all-image mean; nearest/bilinear lower on both sources) |
| E2 | Best interpolation vs trained CNN | FSRNet-L1 vs Bicubic; `paired_vs_bicubic.csv` (bootstrap CI, win rate) | ✅ FSRNet−Bicubic +3.09 dB [+2.58, +3.64], win 100% (n=66); posters +5.6 dB |
| E3 | Trained CNN vs Real-ESRGAN (fidelity vs perception) | PSNR/SSIM vs LPIPS; `fig_perception_distortion.png` | ✅ FSRNet−Real-ESRGAN: posters +5.51 dB [+4.26, +7.02], win 100%, DIV2K +3.78 dB [+3.39, +4.20], win 100%; LPIPS: RE better on DIV2K (0.192 vs 0.207), FSRNet better on posters (0.068 vs 0.116) |
| E4 | 3× vs 6× strategies | x6 rows: Bicubic 6×, FSRNet 3×→bicubic 2×, FSRNet 3×→3×→↓1.5, Real-ESRGAN 4×→1.5× | ✅ every method −3 to −5 dB from 3× to 6×; FSRNet 3×→3×→↓ vs 3×→bicubic +0.09 dB [+0.03, +0.16], win 79% |
| E5 | Degradation robustness | every method on `bicubic` and `mixed` LR test sets | ✅ see A2; RE perceptually best on mixed inputs (LPIPS 0.192 vs 0.265) |
| E6 | Category-wise performance | `summary_by_category.csv`, `fig_category_gain_*.png` (poster test: 16 images; photographic mainly from DIV2K) | ✅ `summary_by_category.csv` / report Table 3 |
| E7 | Detail / ROI analysis | `roi_metrics.csv`, `visuals/*_roi_*.png` — manual `text_logo` (8 posters), `text_small` (5), `face` (6); automatic edges/texture/flat (all) and face (DIV2K only) | ✅ small text: FSRNet +7.8 dB vs bicubic; RE overshoot 2.3–7.1× FSRNet's |
| E8 | Runtime & memory | `--runtime` → `runtime.csv` (320×180 and 1280×720 → 4K/8K) | ✅ T4, 720p→4K: FSRNet 0.48 s / 293 MB; RE 12.77 s / 719 MB; SRCNN 5292 MB |
| A1 | L1 vs MSE loss | `train.loss=mse` | ✅ L1−MSE +0.08 dB [+0.05, +0.12], win 82% |
| A2 | Train on bicubic vs mixed degradation | `data.degradation=mixed`, evaluated on both LR sets (2×2 table) | ✅ mixed−clean-trained: on mixed inputs +0.80 dB [+0.58, +1.03], win 88%; on clean inputs -0.95 dB [-1.18, -0.75], win 0% |
| A3 | DIV2K-only vs DIV2K + posters training | `"data.include_sources=[div2k,poster]"` → `FSRNet-L1+posters` | ✅ +posters − DIV2K-only on posters +0.11 dB [+0.06, +0.16], win 94%; DIV2K -0.00 dB |
| (opt) | SRCNN vs FSRNet (architecture) | `configs/train_srcnn_x3.yaml` | ✅ FSRNet−SRCNN +1.06 dB [+0.84, +1.31], win 100% |
| (opt) | depth 4 vs 8 vs 16 blocks | `model.n_blocks=4` | not planned (1-week scope) |

## Commands (Colab, see `notebooks/colab_train_evaluate.ipynb`)
```
python scripts/prepare_dataset.py --config configs/dataset.yaml
python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto
python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto train.loss=mse output_dir=outputs/runs/fsrnet_x3_mse_bicubic
python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto data.degradation=mixed output_dir=outputs/runs/fsrnet_x3_l1_mixed
python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto "data.include_sources=[div2k,poster]" output_dir=outputs/runs/fsrnet_x3_l1_bicubic_posters
python scripts/train_model.py --config configs/train_srcnn_x3.yaml --resume auto
python scripts/download_weights.py
python scripts/evaluate_model.py --config configs/eval.yaml --runtime --real-world
```

## Protocol notes
* Test set fixed before training: DIV2K-valid test half (50 photos) + user posters. Validation half used only for checkpoint selection.
* Same LR inputs, crop border and Y-channel convention for every method.
* The benchmark is resumable: re-running `evaluate_model.py` skips (image, method, scale, degradation) results already in `metrics_per_image.csv` (`resume: false` starts over).
* Runtime excludes model loading; median of 3 repeats after 1 warm-up; GPU memory = `torch.cuda.max_memory_allocated`.
* Real-world images: 74 AI-upscaled posters, downscaled ×4 (≤1280 px) and super-resolved ×3; grids show the supplied commercial upscale as a labelled reference (not ground truth). No scores.
