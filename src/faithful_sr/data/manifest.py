"""Dataset manifest: one row per *source* image, with provenance and split.

The manifest is the single source of truth for which image belongs to which
split.  Splits are assigned per source image *before* any patch extraction or
degradation, which makes train/test leakage impossible by construction (and
:func:`check_leakage` verifies it, including near-duplicate detection).
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from faithful_sr.utils.image_io import SUPPORTED_EXTENSIONS, ImageLoadError

CATEGORIES = (
    "poster",          # movie poster / alternative poster
    "illustration",    # digital illustration / painted artwork
    "photographic",    # photographic artwork / portrait / natural photo
    "text_logo",       # typography or logo prominent
    "fine_lines",      # thin lines / geometric graphic elements
    "texture",         # dense texture / grain / background detail
)
SPLITS = ("train", "val", "test", "real_world", "excluded")


@dataclass
class ManifestRow:
    image_id: str
    source: str
    rel_path: str
    width: int
    height: int
    sha1: str
    ahash: str
    categories: str          # ';'-separated subset of CATEGORIES
    split: str
    license: str
    has_gt: bool
    notes: str = ""

    @property
    def category_list(self) -> list[str]:
        return [c for c in self.categories.split(";") if c]


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(pil: Image.Image, size: int = 8) -> str:
    """64-bit average hash; Hamming distance <= 5 flags a likely duplicate."""
    g = np.asarray(pil.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    bits = (g > g.mean()).ravel()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def read_tags(tags_file: Optional[Path]) -> dict[str, dict]:
    """Read ``filename,categories,license,notes`` CSV (categories ';'-separated)."""
    if tags_file is None or not Path(tags_file).exists():
        return {}
    out = {}
    with open(tags_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cats = [c.strip() for c in row.get("categories", "").split(";") if c.strip()]
            bad = [c for c in cats if c not in CATEGORIES]
            if bad:
                raise ValueError(f"{tags_file}: unknown categories {bad} for {row['filename']}; "
                                 f"allowed {CATEGORIES}")
            out[row["filename"].strip()] = {"categories": cats, "license": row.get("license", "").strip(),
                                             "notes": row.get("notes", "").strip()}
    return out


def scan_source(name: str, directory: Path, root: Path, split: str, license: str = "",
                categories: Iterable[str] = (), has_gt: bool = True, tags_file: Optional[Path] = None,
                min_size: int = 96, log=print) -> list[ManifestRow]:
    """Create manifest rows for every readable image in ``directory`` (recursive)."""
    directory = Path(directory)
    if not directory.exists():
        log(f"[manifest] source '{name}': directory {directory} not found - skipped")
        return []
    tags = read_tags(tags_file)
    rows = []
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS or p.name.startswith("."):
            continue
        try:
            with Image.open(p) as im:
                im.load()
                w, h = im.size
                ah = average_hash(im)
        except Exception as e:  # corrupt file: record and exclude, never crash
            log(f"[manifest] corrupt/unreadable {p}: {e}")
            rows.append(ManifestRow(f"{name}_{p.stem}", name, str(p.relative_to(root)), 0, 0, "", "0" * 16,
                                    "", "excluded", license, False, f"unreadable: {e}"))
            continue
        t = tags.get(p.name, {})
        cats = t.get("categories") or list(categories)
        row = ManifestRow(
            image_id=f"{name}_{p.stem}", source=name, rel_path=str(p.relative_to(root)).replace("\\", "/"),
            width=w, height=h, sha1=file_sha1(p), ahash=ah, categories=";".join(cats),
            split=split, license=t.get("license") or license, has_gt=has_gt, notes=t.get("notes", ""))
        if min(w, h) < min_size:
            row.split, row.notes = "excluded", f"smaller than min_size={min_size}"
        rows.append(row)
    log(f"[manifest] source '{name}': {len(rows)} images from {directory}")
    return rows


def assign_auto_splits(rows: list[ManifestRow], ratios: dict[str, float], seed: int) -> None:
    """Randomly split rows whose split is 'auto' according to ``ratios`` (by source image)."""
    auto = [r for r in rows if r.split == "auto"]
    if not auto:
        return
    order = np.random.default_rng(seed).permutation(len(auto))
    names, probs = list(ratios), np.array(list(ratios.values()), dtype=float)
    bounds = np.round(np.cumsum(probs / probs.sum()) * len(auto)).astype(int)
    start = 0
    for name, end in zip(names, bounds):
        for i in order[start:end]:
            auto[i].split = name
        start = end


def mark_duplicates(rows: list[ManifestRow], max_hamming: int = 3, log=print) -> int:
    """Exclude exact duplicates (sha1) and flag near duplicates across splits."""
    seen: dict[str, ManifestRow] = {}
    n = 0
    for r in rows:
        if r.split == "excluded":
            continue
        if r.sha1 in seen:
            r.split, r.notes = "excluded", f"exact duplicate of {seen[r.sha1].image_id}"
            n += 1
        else:
            seen[r.sha1] = r
    active = [r for r in rows if r.split != "excluded"]
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            if a.split != b.split and hamming(a.ahash, b.ahash) <= max_hamming:
                # keep the evaluation copy, exclude the training one
                victim = b if b.split == "train" else a if a.split == "train" else b
                other = a if victim is b else b
                if victim.split != "excluded":
                    victim.split, victim.notes = "excluded", f"near-duplicate of {other.image_id}"
                    log(f"[manifest] near-duplicate {victim.image_id} ~ {other.image_id}: excluded")
                    n += 1
    return n


def check_leakage(rows: list[ManifestRow]) -> None:
    """Raise if any source image (id, file hash) appears in more than one split."""
    by_id: dict[str, str] = {}
    by_hash: dict[str, str] = {}
    for r in rows:
        if r.split == "excluded":
            continue
        for key, table in ((r.image_id, by_id), (r.sha1, by_hash)):
            if key in table and table[key] != r.split:
                raise AssertionError(f"Leakage: {r.image_id} in both {table[key]} and {r.split}")
            table[key] = r.split


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[x.name for x in fields(ManifestRow)])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def read_manifest(path: Path) -> list[ManifestRow]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Manifest {path} not found - run scripts/prepare_dataset.py first")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            d["width"], d["height"] = int(d["width"]), int(d["height"])
            d["has_gt"] = str(d["has_gt"]).lower() == "true"
            rows.append(ManifestRow(**d))
    return rows


def rows_for_split(rows: list[ManifestRow], split: str) -> list[ManifestRow]:
    return [r for r in rows if r.split == split]


def load_hr(row: ManifestRow, root: Path) -> np.ndarray:
    from faithful_sr.utils.image_io import load_image
    p = Path(root) / row.rel_path
    try:
        return load_image(p).rgb
    except ImageLoadError:
        raise
