"""Idempotent Colab data setup: safe to re-run after any disconnect or restart.

python scripts/colab_setup.py --drive /content/drive/MyDrive/cs663

First session : download DIV2K (cached on Drive), copy posters from Drive, run
                prepare_dataset, then save data/processed as ONE tar file on Drive.
Later sessions: just unpack that tar (a couple of minutes) - no re-download, no re-prepare.

The cache file name contains a hash of everything that affects the prepared data
(dataset/degradation configs, poster tags, data-pipeline code), so changing any of
them automatically triggers a fresh prepare instead of silently reusing stale data.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/{}.zip"
KEY_FILES = ["configs/dataset.yaml", "configs/degradation.yaml", "data/poster_tags.csv",
             "src/faithful_sr/data/prepare.py", "src/faithful_sr/data/degradation.py",
             "src/faithful_sr/data/manifest.py", "src/faithful_sr/data/patches.py",
             "src/faithful_sr/models/interpolation.py", "src/faithful_sr/utils/image_io.py"]


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=ROOT)


def cache_key() -> str:
    h = hashlib.sha1()
    for f in KEY_FILES:
        p = ROOT / f
        h.update(f.encode())
        h.update(p.read_bytes() if p.exists() else b"missing")
    return h.hexdigest()[:10]


def ensure_div2k(drive: Path) -> None:
    cache = drive / "data_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for part in ("DIV2K_train_HR", "DIV2K_valid_HR"):
        z = cache / f"{part}.zip"
        if not z.exists():
            sh(f'wget -q --show-progress -O "{z}.part" {DIV2K_URL.format(part)} && mv "{z}.part" "{z}"')
        target = ROOT / "data" / "raw" / part
        if not (target.exists() and len(list(target.glob("*.png"))) >= (800 if "train" in part else 100)):
            sh(f'mkdir -p data/raw && unzip -q -o "{z}" -d data/raw/')


def copy_user_data(drive: Path) -> None:
    for src, dst in (("posters", "data/raw/posters"), ("posters_upscaled", "data/raw/posters_upscaled"),
                     ("real_world", "data/real_world")):
        s, d = drive / src, ROOT / dst
        d.mkdir(parents=True, exist_ok=True)
        if not s.exists():
            print(f"[setup] {s} not found on Drive - skipped")
            continue
        n_new = 0
        for f in s.iterdir():
            if f.is_file() and not (d / f.name).exists():
                shutil.copy2(f, d / f.name)
                n_new += 1
        print(f"[setup] {dst}: {len(list(d.iterdir()))} files ({n_new} newly copied)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default="/content/drive/MyDrive/cs663")
    ap.add_argument("--force-prepare", action="store_true", help="ignore the Drive cache and prepare again")
    a = ap.parse_args(argv)
    drive = Path(a.drive)
    if not drive.exists():
        print(f"Drive folder {drive} not found - run cell 1 (mount Drive) and check MyDrive/cs663 exists.")
        return 1
    key = cache_key()
    tar = drive / "data_cache" / f"processed_{key}.tar"
    done_flag = ROOT / "data" / "processed" / f".ready_{key}"
    t0 = time.time()

    if done_flag.exists() and not a.force_prepare:
        print(f"[setup] data/processed already ready in this session (cache key {key}).")
    elif tar.exists() and not a.force_prepare:
        print(f"[setup] unpacking cached prepared dataset {tar.name} from Drive ...")
        sh("rm -rf data/processed && mkdir -p data")
        sh(f'tar -xf "{tar}" -C data')
        done_flag.touch()
    else:
        print(f"[setup] no cache for key {key}: downloading/copying data and preparing (first time ~30-40 min)")
        ensure_div2k(drive)
        copy_user_data(drive)
        sh("python scripts/prepare_dataset.py --config configs/dataset.yaml")
        print(f"[setup] saving prepared dataset to Drive as {tar.name} (one file, a few GB) ...")
        sh(f'tar -cf "{tar}.part" -C data processed && mv "{tar}.part" "{tar}"')
        old = [p for p in (drive / "data_cache").glob("processed_*.tar") if p != tar]
        for p in old:
            print(f"[setup] note: older cache {p.name} can be deleted from Drive to free space")
        done_flag.touch()

    summ = ROOT / "data" / "processed" / "dataset_summary.json"
    if summ.exists():
        import json
        s = json.loads(summ.read_text())
        print("[setup] split counts:", s.get("split_counts"), "| train sub-images:", s.get("train_subimages"))
    print(f"[setup] done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
