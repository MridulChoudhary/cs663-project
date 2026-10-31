"""Generates notebooks/colab_train_evaluate.ipynb (keeps the notebook diff-friendly to edit as code)."""
import json
from pathlib import Path

REPO = "https://github.com/MridulChoudhary/cs663-project.git"

cells = [
    ("md", """# CS663 — Faithful SR: train & evaluate on Colab

Runtime → Change runtime type → **T4 GPU**. Every step is restartable: data, checkpoints and results
live on Google Drive (`MyDrive/cs663`), and training resumes exactly with `--resume auto` after a disconnect.

**Before running:** upload to Google Drive `MyDrive/cs663/`:
* `posters/` — the 54 genuine hi-res originals (tags and ROI boxes come with the code: `data/poster_tags.csv`, `data/rois.json`)
* `posters_upscaled/` — the AI-upscaled files (real-world qualitative set, never ground truth)
"""),
    ("code", """# 1. Drive + code
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/cs663'
!mkdir -p $DRIVE/outputs $DRIVE/models $DRIVE/data_cache $DRIVE/posters $DRIVE/posters_upscaled $DRIVE/real_world
%cd /content
!test -d cs663-project || git clone REPO_URL cs663-project
%cd /content/cs663-project
!git pull --ff-only
# persist outputs/ and models/ on Drive
!rm -rf outputs models && ln -s $DRIVE/outputs outputs && ln -s $DRIVE/models models
!pip -q install -r requirements.txt -r requirements-pretrained.txt
!nvidia-smi --query-gpu=name,memory.total --format=csv""".replace("REPO_URL", REPO)),
    ("code", """# 2. DIV2K HR (train 800 + valid 100), cached on Drive after the first download (~3.5 GB)
import os
for part in ['DIV2K_train_HR', 'DIV2K_valid_HR']:
    z = f'{DRIVE}/data_cache/{part}.zip'
    if not os.path.exists(z):
        !wget -q --show-progress -O {z}.part http://data.vision.ee.ethz.ch/cvl/DIV2K/{part}.zip && mv {z}.part {z}
    if not os.path.exists(f'data/raw/{part}'):
        !mkdir -p data/raw && unzip -q {z} -d data/raw/
# user data
!mkdir -p data/raw/posters data/raw/posters_upscaled data/real_world
!cp -rn $DRIVE/posters/. data/raw/posters/ ; cp -rn $DRIVE/posters_upscaled/. data/raw/posters_upscaled/ ; cp -rn $DRIVE/real_world/. data/real_world/ ; cp -n $DRIVE/rois.json data/rois.json 2>/dev/null; true
!ls data/raw/DIV2K_train_HR | wc -l; ls data/raw/posters | wc -l; ls data/raw/posters_upscaled | wc -l"""),
    ("code", """# 3. Prepare dataset (manifest, splits, train sub-images, fixed val/test LR sets) - ~10 min first time
!python scripts/prepare_dataset.py --config configs/dataset.yaml
!cat data/processed/dataset_summary.json | head -40"""),
    ("code", """# 4. Tests + smoke run (should take ~1 minute)
!python scripts/make_sample_data.py && python scripts/prepare_dataset.py --config configs/sample_dataset.yaml > /dev/null
!python -m pytest -q
!python scripts/train_model.py --config configs/smoke.yaml"""),
    ("md", """## 5. Training runs
Re-run a cell after a disconnect: `--resume auto` continues from `outputs/runs/<name>/latest.pt`.
Order of priority (1-week plan): main model → MSE ablation → mixed-degradation ablation → posters ablation (A3) → SRCNN."""),
    ("code", """# 5a. Main model: FSRNet x3, L1, bicubic degradation
!python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto
!cp outputs/runs/fsrnet_x3_l1_bicubic/best.pt models/fsrnet_x3_l1_bicubic.pt"""),
    ("code", """# 5b. Ablation A1: MSE loss (everything else identical)
!python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto train.loss=mse output_dir=outputs/runs/fsrnet_x3_mse_bicubic
!cp outputs/runs/fsrnet_x3_mse_bicubic/best.pt models/fsrnet_x3_mse_bicubic.pt"""),
    ("code", """# 5c. Ablation A2: train on the mixed (blur+noise+JPEG) degradation
!python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto data.degradation=mixed output_dir=outputs/runs/fsrnet_x3_l1_mixed
!cp outputs/runs/fsrnet_x3_l1_mixed/best.pt models/fsrnet_x3_l1_mixed.pt"""),
    ("code", """# 5d. Ablation A3: DIV2K + poster training images (posters' own train split)
!python scripts/train_model.py --config configs/train_fsrnet_x3.yaml --resume auto "data.include_sources=[div2k,poster]" output_dir=outputs/runs/fsrnet_x3_l1_bicubic_posters
!cp outputs/runs/fsrnet_x3_l1_bicubic_posters/best.pt models/fsrnet_x3_l1_bicubic_posters.pt"""),
    ("code", """# 5e. SRCNN paper baseline
!python scripts/train_model.py --config configs/train_srcnn_x3.yaml --resume auto
!cp outputs/runs/srcnn_x3_l1_bicubic/best.pt models/srcnn_x3_l1_bicubic.pt"""),
    ("code", """# Training curves
from IPython.display import Image, display
import glob
for f in sorted(glob.glob('outputs/runs/*/curves.png')):
    print(f); display(Image(f))"""),
    ("code", """# 6. Pretrained comparator weights (explicit download) + full benchmark (E1-E8)
!python scripts/download_weights.py
!python scripts/evaluate_model.py --config configs/eval.yaml --runtime --real-world
!cat outputs/eval/main/status.csv"""),
    ("code", """# 7. Results
import pandas as pd
display(pd.read_csv('outputs/eval/main/report/summary_by_method.csv').round(3))
display(pd.read_csv('outputs/eval/main/report/paired_vs_bicubic.csv').round(3))
for f in sorted(glob.glob('outputs/eval/main/report/fig_*.png')):
    print(f); display(Image(f))"""),
    ("code", """# 8. (optional) GUI with a temporary public link
!python scripts/launch_gui.py --share"""),
]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": "markdown" if t == "md" else "code", "metadata": {},
                 "source": s, **({} if t == "md" else {"outputs": [], "execution_count": None})}
                for t, s in cells]}
Path(__file__).with_name("colab_train_evaluate.ipynb").write_text(json.dumps(nb, indent=1))
print("written")
