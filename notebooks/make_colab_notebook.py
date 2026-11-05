"""Generates notebooks/colab_train_evaluate.ipynb (keeps the notebook diff-friendly to edit as code).

Design rule: every code cell is self-contained (re-enters the project folder and re-defines
DRIVE), so after a Colab disconnect/restart you can re-run cell 1 and then jump straight to
the cell you need - no hidden state from earlier cells.
"""
import json
from pathlib import Path

REPO = "https://github.com/MridulChoudhary/cs663-project.git"
HDR = "%cd /content/cs663-project\nDRIVE = '/content/drive/MyDrive/cs663'\n"


def train_cell(title, args, run, model_name):
    return ("code", f"""# {title}
{HDR}# the model is copied to models/ ONLY if training completes without error (single shell line with &&)
!python scripts/train_model.py --config {args} --resume auto && cp outputs/runs/{run}/best.pt models/{model_name}.pt && echo "FINISHED - saved models/{model_name}.pt\"""")


cells = [
    ("md", """# CS663 — Faithful SR: train & evaluate on Colab

Runtime → Change runtime type → **T4 GPU**. Run cells **one at a time** (Shift+Enter), not "Run all".

**After any disconnect / restart:** run **cell 1**, then **cell 2** (it only unpacks the cached dataset,
a few minutes), then the training cell you were on — training resumes from its last checkpoint.

Drive layout expected in `MyDrive/cs663/` (same Google account as this Colab):
* `posters/` — the 54 genuine hi-res originals
* `posters_upscaled/` — the AI-upscaled files (real-world qualitative set, never ground truth)

Tags and ROI boxes come with the code (`data/poster_tags.csv`, `data/rois.json`).
Everything the notebook produces is kept on Drive: `outputs/`, `models/`, `data_cache/`."""),
    ("code", f"""# 1. Drive + code + packages  (re-run first after every disconnect/restart)
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/cs663'
for d in ('outputs', 'models', 'data_cache'):
    os.makedirs(f'{{DRIVE}}/{{d}}', exist_ok=True)
%cd /content
if not os.path.isdir('/content/cs663-project/.git'):
    !git clone {REPO}
%cd /content/cs663-project
!git pull --ff-only
# outputs/ and models/ are shortcuts (symlinks) to Drive; rm only ever removes the shortcut, never Drive files
for d in ('outputs', 'models'):
    if not os.path.islink(d):
        !rm -rf {{d}} && ln -s {{DRIVE}}/{{d}} {{d}}
!ls -l outputs models | head -3
!pip -q install -r requirements.txt -r requirements-pretrained.txt
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", f"""# 2. Data: first session downloads DIV2K + copies posters + prepares (~30-40 min) and saves ONE
#    cache file to Drive; every later session just unpacks it (a few minutes). Safe to re-run.
{HDR}!python scripts/colab_setup.py --drive {{DRIVE}}"""),
    ("code", f"""# 3. (optional) Tests + smoke run on the GPU machine (~2 minutes)
{HDR}!python scripts/make_sample_data.py && python scripts/prepare_dataset.py --config configs/sample_dataset.yaml > /dev/null
!python -m pytest -q
!python scripts/train_model.py --config configs/smoke.yaml"""),
    ("md", """## 4. Training runs
Each cell resumes automatically (`--resume auto`) if it was interrupted.
Priority (1-week plan): 4a main model → 4b MSE → 4c mixed degradation → 4d posters (A3) → 4e SRCNN."""),
    train_cell("4a. Main model: FSRNet x3, L1, bicubic degradation, DIV2K only",
               "configs/train_fsrnet_x3.yaml", "fsrnet_x3_l1_bicubic", "fsrnet_x3_l1_bicubic"),
    train_cell("4b. Ablation A1: MSE loss (everything else identical)",
               "configs/train_fsrnet_x3.yaml train.loss=mse output_dir=outputs/runs/fsrnet_x3_mse_bicubic",
               "fsrnet_x3_mse_bicubic", "fsrnet_x3_mse_bicubic"),
    train_cell("4c. Ablation A2: train on the mixed (blur+noise+JPEG) degradation",
               "configs/train_fsrnet_x3.yaml data.degradation=mixed output_dir=outputs/runs/fsrnet_x3_l1_mixed",
               "fsrnet_x3_l1_mixed", "fsrnet_x3_l1_mixed"),
    train_cell("4d. Ablation A3: DIV2K + poster training images (posters' own train split)",
               'configs/train_fsrnet_x3.yaml "data.include_sources=[div2k,poster]" output_dir=outputs/runs/fsrnet_x3_l1_bicubic_posters',
               "fsrnet_x3_l1_bicubic_posters", "fsrnet_x3_l1_bicubic_posters"),
    train_cell("4e. SRCNN paper baseline", "configs/train_srcnn_x3.yaml",
               "srcnn_x3_l1_bicubic", "srcnn_x3_l1_bicubic"),
    ("code", f"""# Training curves + where each run stands
{HDR}from IPython.display import Image, display
import glob
!tail -n 2 outputs/runs/*/val_log.csv 2>/dev/null
for f in sorted(glob.glob('outputs/runs/*/curves.png')):
    print(f); display(Image(f))"""),
    ("code", f"""# 5. Pretrained comparator weights (explicit download) + full benchmark (E1-E8), ~1.5-2 h.
#    Resumable: after a disconnect run cells 1, 2 and this cell again - finished results are skipped.
#    Needs cell 2 done in this session. Methods without a checkpoint are reported as 'not run'.
{HDR}!python scripts/download_weights.py
!python scripts/evaluate_model.py --config configs/eval.yaml --runtime --real-world
!cat outputs/eval/main/status.csv"""),
    ("code", f"""# 6. Results
{HDR}import pandas as pd, glob
from IPython.display import Image, display
display(pd.read_csv('outputs/eval/main/report/summary_by_method.csv').round(3))
display(pd.read_csv('outputs/eval/main/report/paired_vs_bicubic.csv').round(3))
for f in sorted(glob.glob('outputs/eval/main/report/fig_*.png')):
    print(f); display(Image(f))"""),
    ("code", f"""# 7. (optional) GUI with a temporary public link - this cell keeps running until you stop it
{HDR}!python scripts/launch_gui.py --share"""),
]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": "markdown" if t == "md" else "code", "metadata": {},
                 "source": s, **({} if t == "md" else {"outputs": [], "execution_count": None})}
                for t, s in cells]}
Path(__file__).with_name("colab_train_evaluate.ipynb").write_text(json.dumps(nb, indent=1))
print("written")
