# Faithful Image Super-Resolution for Movie Posters and Digital Artwork
*CS663 Digital Image Processing — course project*

How do classical interpolation, a CNN we implement and train ourselves, and a pretrained perceptual SR model
(Real-ESRGAN) compare in **reconstruction fidelity** and **visual quality** on posters and artwork, and how do
the degradation model and the scale factor (3× → 4K, 6× → 8K) change that picture?

> Super-resolution estimates missing detail. Sharper is not the same as more accurate: outputs are compared
> against real high-resolution ground truth wherever it exists, and real-world images without ground truth
> are evaluated qualitatively only.

## What is implemented
| Component | Where | Notes |
|---|---|---|
| Nearest / bilinear / bicubic / Lanczos **from scratch** | `src/faithful_sr/models/interpolation.py` | sparse separable kernels, anti-aliased downscaling, verified vs Pillow |
| Degradation model (blur → ↓s → noise → 8-bit → JPEG) | `src/faithful_sr/data/degradation.py` | presets `bicubic, blur, noise, jpeg, mixed`; deterministic; alignment-tested |
| Manifest, source-level splits, duplicate/leakage checks | `src/faithful_sr/data/manifest.py`, `prepare.py` | contact sheet + summary + log |
| SRCNN and **FSRNet** (residual + sub-pixel, ours) | `src/faithful_sr/models/cnn.py` | ≈0.37 M params |
| Training with exact resume, best/latest checkpoints | `src/faithful_sr/training/` | L1/MSE, cosine LR, Colab-safe |
| Real-ESRGAN x4plus adapter (pretrained comparator) | `src/faithful_sr/inference/learned.py` | via `spandrel`, explicit download |
| 6× strategies + seam-free tiled inference | `inference/learned.py`, `inference/tiling.py` | tiled == whole image (< 1e-5) |
| PSNR/SSIM (own, = scikit-image), LPIPS, edge-PSNR, overshoot | `src/faithful_sr/evaluation/metrics.py` | Y channel BT.601, border crop = scale |
| Benchmark, category & ROI analysis, runtime, reports/plots | `src/faithful_sr/evaluation/` | not-run methods reported, never skipped |
| Gradio GUI (upload, method, 3×/6×, 4K/8K pad/crop, slider, save) | `src/faithful_sr/gui/app.py` | thin layer over the CLI API |

Maths and design rationale: [`docs/methodology.md`](docs/methodology.md). Experiment plan/status:
[`docs/experiments.md`](docs/experiments.md). Data and licences: [`data/README.md`](data/README.md).

## Setup
```bash
git clone https://github.com/MridulChoudhary/cs663-project.git && cd cs663-project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-pretrained.txt             # optional: Real-ESRGAN + LPIPS
python scripts/download_weights.py                     # optional: explicit download of third-party weights
```

## Quick check (CPU, ~1 minute, no downloads)
```bash
python scripts/make_sample_data.py
python scripts/prepare_dataset.py --config configs/sample_dataset.yaml
python -m pytest -q
python scripts/train_model.py --config configs/smoke.yaml
python scripts/evaluate_model.py --config configs/eval_smoke.yaml --runtime --real-world
```

## Full pipeline (GPU — use `notebooks/colab_train_evaluate.ipynb` on Colab)
On Colab, after any disconnect: re-run cell 1, then cell 2 (`scripts/colab_setup.py` unpacks the cached prepared dataset from Drive), then the cell you were on.

```bash
python scripts/prepare_dataset.py --config configs/dataset.yaml
python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto
python scripts/evaluate_model.py --config configs/eval.yaml --runtime --real-world
```

## Upscale images / GUI
```bash
python scripts/upscale.py poster.png --method bicubic --scale 3
python scripts/upscale.py poster.png --method cnn --checkpoint models/fsrnet_x3_l1_bicubic.pt --scale 6 --tile 256
python scripts/launch_gui.py            # http://127.0.0.1:7860   (--share for a public link on Colab)
```

## Licences and citations
Code: project code by the CS663 team. DIV2K: academic use only, not redistributed. Real-ESRGAN weights:
BSD-3-Clause (X. Wang et al., *Real-ESRGAN*, ICCVW 2021, https://github.com/xinntao/Real-ESRGAN), loaded with
`spandrel` (MIT). LPIPS: BSD-2 (Zhang et al., CVPR 2018). User-supplied posters are evaluated privately and
not redistributed.
