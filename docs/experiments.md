# Experiments — plan, commands and status

Status legend: ✅ done (results saved) · ⏳ pending (not yet run) · ⚙️ pipeline verified on synthetic sample only.
**No result below is a real number until its status is ✅** — values come only from `outputs/eval/main/report/`.

| ID | Question | How (config / command) | Status |
|---|---|---|---|
| E1 | Nearest vs bilinear vs bicubic | `configs/eval.yaml` methods 1–3 | ⚙️ |
| E2 | Best interpolation vs trained CNN | FSRNet-L1 vs Bicubic; `paired_vs_bicubic.csv` (bootstrap CI, win rate) | ⏳ train on Colab |
| E3 | Trained CNN vs Real-ESRGAN (fidelity vs perception) | PSNR/SSIM vs LPIPS; `fig_perception_distortion.png` | ⏳ |
| E4 | 3× vs 6× strategies | x6 rows: Bicubic 6×, FSRNet 3×→bicubic 2×, FSRNet 3×→3×→↓1.5, Real-ESRGAN 4×→1.5× | ⏳ |
| E5 | Degradation robustness | every method on `bicubic` and `mixed` LR test sets | ⏳ |
| E6 | Category-wise performance | `summary_by_category.csv`, `fig_category_gain_*.png` (poster test: 16 images; photographic mainly from DIV2K) | ⏳ data ready |
| E7 | Detail / ROI analysis | `roi_metrics.csv`, `visuals/*_roi_*.png` — manual `text_logo` (8 posters), `text_small` (5), `face` (6); automatic edges/texture/flat (all) and face (DIV2K only) | ⏳ ROIs ready |
| E8 | Runtime & memory | `--runtime` → `runtime.csv` (320×180 and 1280×720 → 4K/8K) | ⏳ GPU |
| A1 | L1 vs MSE loss | `train.loss=mse` | ⏳ |
| A2 | Train on bicubic vs mixed degradation | `data.degradation=mixed`, evaluated on both LR sets (2×2 table) | ⏳ |
| A3 | DIV2K-only vs DIV2K + posters training | `"data.include_sources=[div2k,poster]"` → `FSRNet-L1+posters` | ⏳ |
| (opt) | SRCNN vs FSRNet (architecture) | `configs/train_srcnn_x3.yaml` | ⏳ |
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
* Runtime excludes model loading; median of 3 repeats after 1 warm-up; GPU memory = `torch.cuda.max_memory_allocated`.
* Real-world images: 74 AI-upscaled posters, downscaled ×4 (≤1280 px) and super-resolved ×3; grids show the supplied commercial upscale as a labelled reference (not ground truth). No scores.
