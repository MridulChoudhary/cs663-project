"""Regions of interest (ROIs) for detail-preservation analysis.

ROI sources, in priority order (all repeatable):

1. **Manual** boxes in a JSON file (``data/rois.json``)::

       {"poster_film01": [{"label": "text_logo", "box": [x, y, w, h]}, ...]}

   Coordinates are in the HR ground-truth image *after* mod-cropping
   (``data/processed/test/HR/<id>.png``).  Use this for typography/logos.
2. **Automatic** boxes, computed deterministically from the HR image only
   (never from a method's output, so no method is favoured):

   * ``face``      — largest OpenCV Haar-cascade frontal face (if any). The cascade is
                     trained on photos and fires on stylized art (balls, fire, crystals), so
                     it is disabled for ids matching ``auto_face_exclude_prefixes`` (posters),
                     whose faces are annotated manually instead;
   * ``edges``     — the window with the highest fraction of strong Sobel edges
                     (lines, lettering, graphic elements);
   * ``texture``   — the window with the highest Laplacian energy among windows
                     whose strong-edge fraction is below the median (fine texture
                     rather than isolated edges);
   * ``flat``      — the lowest-variance window (control: all methods should do
                     well here).

All boxes are snapped to multiples of 6 so the same region maps exactly onto
the LR grid at both 3x and 6x.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import laplace

from faithful_sr.evaluation.metrics import edge_psnr, gradient_magnitude, overshoot, psnr, rgb_to_y, ssim

SNAP = 6


def snap_box(box, h: int, w: int, snap: int = SNAP) -> list[int]:
    x, y, bw, bh = (int(v) for v in box)
    x, y = (x // snap) * snap, (y // snap) * snap
    bw, bh = max(snap * 4, (bw // snap) * snap), max(snap * 4, (bh // snap) * snap)
    bw, bh = min(bw, (w // snap) * snap), min(bh, (h // snap) * snap)
    x, y = min(x, w - bw), min(y, h - bh)
    return [x, y, bw, bh]


def _windows(h: int, w: int, size: int, step: int):
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            yield y, x


def auto_rois(hr: np.ndarray, win_frac: float = 0.15, face: bool = True) -> list[dict]:
    h, w = hr.shape[:2]
    size = max(48, int(min(h, w) * win_frac) // SNAP * SNAP)
    step = size // 2
    y_ch = rgb_to_y(hr)
    gm = gradient_magnitude(hr)
    strong = gm > np.quantile(gm, 0.9)
    lap = np.abs(laplace(y_ch))
    stats = []
    for y, x in _windows(h, w, size, step):
        sl = (slice(y, y + size), slice(x, x + size))
        stats.append((y, x, strong[sl].mean(), lap[sl].mean(), y_ch[sl].var()))
    if not stats:
        return []
    arr = np.array(stats)
    rois = []
    i_edge = int(arr[:, 2].argmax())
    rois.append({"label": "edges", "box": [int(arr[i_edge, 1]), int(arr[i_edge, 0]), size, size], "source": "auto"})
    med = np.median(arr[:, 2])
    cand = np.where((arr[:, 2] <= med) & (np.arange(len(arr)) != i_edge))[0]
    if len(cand):
        i_tex = int(cand[arr[cand, 3].argmax()])
        rois.append({"label": "texture", "box": [int(arr[i_tex, 1]), int(arr[i_tex, 0]), size, size], "source": "auto"})
    i_flat = int(arr[:, 4].argmin())
    rois.append({"label": "flat", "box": [int(arr[i_flat, 1]), int(arr[i_flat, 0]), size, size], "source": "auto"})
    fbox = detect_face(hr) if face else None
    if fbox is not None:
        rois.append({"label": "face", "box": fbox, "source": "auto"})
    for r in rois:
        r["box"] = snap_box(r["box"], h, w)
    return rois


def detect_face(hr: np.ndarray) -> Optional[list[int]]:
    try:
        import cv2
    except ImportError:
        return None
    name = "haarcascade_frontalface_default.xml"
    candidates = [Path("models/pretrained") / name, Path(__file__).resolve().parents[3] / "models/pretrained" / name]
    if hasattr(cv2, "data"):
        candidates.insert(0, Path(cv2.data.haarcascades) / name)
    path = next((p for p in candidates if p.exists()), None)
    if path is None:      # OpenCV 5 wheels no longer bundle cascades: scripts/download_weights.py fetches it
        return None
    casc = cv2.CascadeClassifier(str(path))
    if casc.empty():
        return None
    gray = cv2.cvtColor((hr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    m = max(24, min(hr.shape[:2]) // 12)
    faces = casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(m, m))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * w)
    return [max(0, x - pad), max(0, y - pad), w + 2 * pad, h + 2 * pad]


def load_manual_rois(path: Optional[str | Path]) -> dict[str, list[dict]]:
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for iid, lst in data.items():
        for r in lst:
            if "label" not in r or "box" not in r or len(r["box"]) != 4:
                raise ValueError(f"{path}: ROI for {iid} needs 'label' and 'box': [x, y, w, h]")
            r.setdefault("source", "manual")
    return data


def rois_for_image(image_id: str, hr: np.ndarray, manual: dict, auto_face: bool = True) -> list[dict]:
    h, w = hr.shape[:2]
    rois = [dict(r, box=snap_box(r["box"], h, w)) for r in manual.get(image_id, [])]
    manual_labels = {r["label"] for r in rois}
    rois += [r for r in auto_rois(hr, face=auto_face) if r["label"] not in manual_labels]
    return rois


def roi_metrics(sr: np.ndarray, hr: np.ndarray, box) -> dict:
    x, y, w, h = box
    a, b = sr[y:y + h, x:x + w], hr[y:y + h, x:x + w]
    return {"psnr_y": psnr(a, b), "ssim_y": ssim(a, b) if min(w, h) >= 24 else float("nan"),
            "edge_psnr": edge_psnr(a, b), "overshoot": overshoot(a, b)}
