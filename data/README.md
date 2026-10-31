# Data — where to put it (nothing here is committed to git)

| Folder | What | Split | Licence / provenance |
|---|---|---|---|
| `data/raw/DIV2K_train_HR/` | 800 DIV2K HR images (`DIV2K_train_HR.zip`) | train | Agustsson & Timofte, CVPRW 2017 — academic research use only. Download from the official DIV2K page; do not redistribute. |
| `data/raw/DIV2K_valid_HR/` | 100 DIV2K HR validation images | seeded 50/50 → val / test | same |
| `data/raw/posters/` | Genuine hi-res posters / artwork (54 numbered files); tags in committed `data/poster_tags.csv` | seeded 70/30 → train / test | Free-reuse source sites; private academic use; not redistributed. |
| `data/raw/posters_upscaled/` | Posters that were low-res and **AI-upscaled by third-party services** | real_world (never ground truth) | Downscaled ×4 → LR input; the supplied file is shown as a labelled reference only. |
| `data/real_world/` | Naturally low-res images (unknown history) | real_world (no ground truth) | Qualitative comparisons only — no scores are computed. |
| `data/rois.json` | Manual ROI boxes: `text_logo`, `text_small`, `face` on the poster test set (assistant-annotated, author-reviewed) | — | coordinates in `data/processed/test/HR/<id>.png` pixels |

## `tags.csv` format
```
filename,categories,license,notes
film01.png,poster;text_logo;photographic,"own scan of my poster","title block at bottom"
art02.jpg,illustration;fine_lines,"CC BY 4.0 - <artist>, <url>",
```
Allowed categories (an image may have several): `poster`, `illustration`, `photographic`, `text_logo`, `fine_lines`, `texture`.

Use the **highest-resolution** source you have (≥ 1500 px on the short side is ideal): the HR image is the ground truth, and LR inputs are generated from it, so a poster that was itself upscaled or heavily JPEG-compressed makes a poor reference.

## Prepare
```
python scripts/prepare_dataset.py --config configs/dataset.yaml
```
Creates `data/processed/` (manifest, split lists, contact sheet, summary, log, train sub-images, fixed val/test LR sets). Splits are made per **source image** before any patching or degradation; exact and near-duplicates across splits are excluded and logged.

## Tiny synthetic sample (no downloads)
```
python scripts/make_sample_data.py
python scripts/prepare_dataset.py --config configs/sample_dataset.yaml
```
Procedurally generated poster-like images (project-made) + public-domain/CC0 photos bundled with scikit-image.
