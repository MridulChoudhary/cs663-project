"""Gradio GUI - a thin layer over faithful_sr.inference.upscale (no model logic here).

Launch:  python -m faithful_sr.gui.app            (or: python scripts/launch_gui.py)
         python scripts/launch_gui.py --share     (public link, e.g. from Colab)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from faithful_sr.inference.sizing import UHD, fit_canvas
from faithful_sr.inference.upscale import HALLUCINATION_WARNING, available_methods, upscale_array
from faithful_sr.models.interpolation import upscale as classical_upscale
from faithful_sr.utils.image_io import ImageLoadError, load_image, save_image, to_uint8

OUT_DIR = Path("outputs/gui")
PREVIEW_MAX = 2048          # longer side of the on-screen preview (full result is saved to disk)
LABELS = {"nearest": "Nearest neighbour", "bilinear": "Bilinear", "bicubic": "Bicubic",
          "cnn": "Trained CNN (checkpoint)", "realesrgan": "Real-ESRGAN x4plus (pretrained)"}


def find_checkpoints() -> list[str]:
    found = sorted({str(p) for pat in ("models/*.pt", "outputs/runs/*/best.pt") for p in Path(".").glob(pat)})
    return found


def _preview(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, PREVIEW_MAX / max(h, w))
    if s < 1:
        from faithful_sr.models.interpolation import resize
        img = resize(img, round(h * s), round(w * s), "bicubic")
    return to_uint8(img)


def describe(path: Optional[str]):
    if not path:
        return "No image loaded."
    try:
        li = load_image(path)
    except ImageLoadError as e:
        return f"⚠️ {e}"
    lines = [f"**Input:** {li.width} × {li.height} px (mode {li.original_mode})",
             f"3× → {li.width * 3} × {li.height * 3} px · 6× → {li.width * 6} × {li.height * 6} px"]
    lines += [f"⚠️ {w}" for w in li.warnings]
    return "\n\n".join(lines)


def run(path, method_label, scale, checkpoint, canvas, canvas_mode, tile, overlap, six_strategy, fmt, jpeg_q,
        progress=None):
    if not path:
        raise ValueError("Upload an image first.")
    method = {v: k for k, v in LABELS.items()}.get(method_label, method_label)
    scale = int(str(scale).rstrip("×x"))
    li = load_image(path)
    if li.width * li.height * scale * scale > 7680 * 4320 * 1.3:
        raise ValueError(f"Output would be {li.width*scale}×{li.height*scale}; refusing above ~8K. "
                         "Use a smaller input or 3×.")
    kwargs = {}
    if method == "cnn":
        if not checkpoint:
            raise ValueError("No checkpoint selected. Train one (scripts/train_model.py) or place a .pt in models/.")
        kwargs = {"checkpoint": checkpoint, "six_strategy": six_strategy}
    if method in ("cnn", "realesrgan") and int(tile) > 0:
        kwargs.update(tile=int(tile), overlap=int(overlap))
    res = upscale_array(li.rgb, method, scale, li.alpha, **kwargs)
    out, meta = res.image, res.metadata()
    meta["warnings"] = li.warnings + res.warnings
    if canvas != "Keep aspect ratio (input × scale)":
        cw, ch = UHD[canvas]
        out, cinfo = fit_canvas(out, cw, ch, "pad" if canvas_mode.startswith("Pad") else "crop")
        meta.update(cinfo)
        meta["output_size"] = (cw, ch)
        meta["warnings"].append(f"Resized uniformly by {cinfo['resize_factor']} with bicubic and "
                                f"{cinfo['canvas_mode']}ed to {cw}×{ch} (no stretching).")
        alpha = None
    else:
        alpha = res.alpha
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    ext = "jpg" if fmt == "JPEG" else "png"
    dst = OUT_DIR / f"{stem}_{method}_x{scale}_{time.strftime('%Y%m%d-%H%M%S')}.{ext}"
    save_image(out, dst, alpha, jpeg_quality=int(jpeg_q), overwrite_protect=path)
    meta["saved_to"] = str(dst)
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    ref = classical_upscale(li.rgb, scale, "nearest")
    if ref.shape != out.shape:   # canvas mode: show the same framing for the reference
        ref, _ = fit_canvas(ref, out.shape[1], out.shape[0], "pad" if canvas_mode.startswith("Pad") else "crop")
    summary = (f"**{LABELS.get(method, method)}** · {scale}× · {res.input_size[0]}×{res.input_size[1]} → "
               f"{meta['output_size'][0]}×{meta['output_size'][1]} · {res.runtime_s:.2f}s on {res.device}")
    if meta["warnings"]:
        summary += "\n\n" + "\n\n".join(f"⚠️ {w}" for w in dict.fromkeys(meta["warnings"]))
    return (_preview(ref), _preview(out)), summary, json.dumps(meta, indent=2, default=str), str(dst)


def build_ui():
    import gradio as gr
    methods = [LABELS.get(m, m) for m in available_methods()]
    ckpts = find_checkpoints()
    with gr.Blocks(title="Faithful SR - CS663") as demo:
        gr.Markdown("# Faithful Super-Resolution for Posters & Artwork\n"
                    f"<small>{HALLUCINATION_WARNING}</small>")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="filepath", label="Input image", sources=["upload", "clipboard"])
                info = gr.Markdown("No image loaded.")
                method = gr.Dropdown(methods, value=LABELS["bicubic"], label="Method")
                scale = gr.Radio(["3×", "6×"], value="3×", label="Scale")
                ckpt = gr.Dropdown(ckpts, value=ckpts[0] if ckpts else None, label="CNN checkpoint",
                                   allow_custom_value=True,
                                   info="Found in models/ and outputs/runs/*/best.pt" if ckpts else
                                   "None found - train a model or copy a .pt into models/")
                canvas = gr.Dropdown(["Keep aspect ratio (input × scale)"] + list(UHD),
                                     value="Keep aspect ratio (input × scale)", label="Output size")
                canvas_mode = gr.Radio(["Pad (letterbox, keep everything)", "Crop (fill canvas)"],
                                       value="Pad (letterbox, keep everything)", label="If aspect ratio differs")
                with gr.Accordion("Advanced", open=False):
                    tile = gr.Slider(0, 1024, value=0, step=32, label="Tile size (LR px, 0 = whole image)")
                    overlap = gr.Slider(0, 64, value=24, step=2, label="Tile overlap (LR px)")
                    six = gr.Radio(["cnn_then_bicubic", "cnn_twice_then_down"], value="cnn_then_bicubic",
                                   label="6× strategy for a 3× CNN")
                    fmt = gr.Radio(["PNG", "JPEG"], value="PNG", label="Save format")
                    jq = gr.Slider(50, 100, value=95, step=1, label="JPEG quality")
                go = gr.Button("Upscale", variant="primary")
            with gr.Column(scale=2):
                slider = gr.ImageSlider(label="Input (nearest) ↔ Output", type="numpy")
                summary = gr.Markdown()
                file_out = gr.File(label="Download full-resolution result")
                meta = gr.Code(label="Metadata", language="json")
        inp.change(describe, inp, info)

        def _run(*a):
            try:
                return run(*a)
            except Exception as e:  # show a readable error instead of a stack trace
                raise gr.Error(str(e))
        go.click(_run, [inp, method, scale, ckpt, canvas, canvas_mode, tile, overlap, six, fmt, jq],
                 [slider, summary, meta, file_out])
    return demo


def main(share: bool = False, port: int = 7860) -> None:
    build_ui().launch(share=share, server_port=port)


if __name__ == "__main__":
    main()
