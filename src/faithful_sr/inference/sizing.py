"""Output sizing for exact UHD targets without ever stretching the image.

Default behaviour is "keep aspect": output = input x scale.  When the user asks
for an exact canvas (3840x2160 / 7680x4320) and the aspect ratio differs, the
SR result is resized uniformly (bicubic) and then either

* ``pad``  - fit entirely inside the canvas, letterbox/pillarbox with a colour, or
* ``crop`` - cover the canvas, centre-crop the overflow.

The uniform resize is a classical step applied *after* super-resolution; the
GUI reports it in the metadata.
"""
from __future__ import annotations

import numpy as np

from faithful_sr.models.interpolation import resize

UHD = {"4K UHD (3840x2160)": (3840, 2160), "8K UHD (7680x4320)": (7680, 4320)}


def fit_canvas(img: np.ndarray, width: int, height: int, mode: str = "pad",
               pad_colour=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img, {"canvas_mode": "exact", "resize_factor": 1.0}
    if mode not in ("pad", "crop"):
        raise ValueError("mode must be 'pad' or 'crop'")
    s = min(width / w, height / h) if mode == "pad" else max(width / w, height / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = resize(img, nh, nw, "bicubic") if (nw, nh) != (w, h) else img
    if mode == "pad":
        out = np.empty((height, width, 3), np.float32)
        out[:] = np.asarray(pad_colour, np.float32)
        y, x = (height - nh) // 2, (width - nw) // 2
        out[y:y + nh, x:x + nw] = r
    else:
        y, x = (nh - height) // 2, (nw - width) // 2
        out = np.ascontiguousarray(r[y:y + height, x:x + width])
    return out, {"canvas_mode": mode, "resize_factor": round(s, 5), "content_size": [nw, nh]}
