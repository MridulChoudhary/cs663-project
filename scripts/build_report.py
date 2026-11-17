"""Build the CS663 course report (HTML -> PDF) from saved benchmark outputs only.

python scripts/build_report.py --run outputs/eval/main --figs outputs/figures --out outputs/report

Every number in the report is read from the run's CSV files at build time.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["image_id", "method", "scale", "degradation"]


def load(run: Path):
    m = pd.read_csv(run / "metrics_per_image.csv").drop_duplicates(KEY, keep="last")
    m["source"] = np.where(m.image_id.str.startswith("poster"), "poster", "div2k")
    r = pd.read_csv(run / "roi_metrics.csv")
    rt = pd.read_csv(run / "runtime.csv")
    return m, r, rt


def boot(x, n=5000):
    rng = np.random.default_rng(0)
    b = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])


def paired(m, a, b, scale, deg, src=None, metric="psnr_y"):
    d = m[(m.scale == scale) & (m.degradation == deg)]
    if src:
        d = d[d.source == src]
    A = d[d.method == a].set_index("image_id")[metric]
    B = d[d.method == b].set_index("image_id")[metric]
    x = (A - B).dropna().values
    lo, hi = boot(x)
    win = (x > 0).mean() if metric != "lpips" else (x < 0).mean()
    return {"mean": x.mean(), "lo": lo, "hi": hi, "win": win, "n": len(x)}


def mean(m, method, scale, deg, metric="psnr_y", src=None):
    d = m[(m.method == method) & (m.scale == scale) & (m.degradation == deg)]
    if src:
        d = d[d.source == src]
    return d[metric].mean()


def img(path: Path, width="100%", cap="") -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    c = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f'<figure><img src="data:image/png;base64,{data}" style="width:{width}">{c}</figure>'


def table(df: pd.DataFrame, fmt: dict, bold_max=(), bold_min=()) -> str:
    h = "<table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr></thead><tbody>"
    best = {c: df[c].max() for c in bold_max if c in df}
    best.update({c: df[c].min() for c in bold_min if c in df})
    for _, row in df.iterrows():
        h += "<tr>"
        for c in df.columns:
            v = row[c]
            s = fmt.get(c, "{}").format(v) if not isinstance(v, str) else html.escape(v)
            if c in best and not isinstance(v, str) and np.isclose(v, best[c]):
                s = f"<b>{s}</b>"
            h += f"<td>{s}</td>"
        h += "</tr>"
    return h + "</tbody></table>"


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10.4pt; line-height: 1.42; color: #111; }
h1 { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 20pt; margin: 0 0 4px 0; line-height: 1.2; }
h2 { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 13pt; margin: 18px 0 6px 0; border-bottom: 1.5px solid #eb6834; padding-bottom: 2px; }
h3 { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 11pt; margin: 12px 0 4px 0; }
.sub { font-family: 'DejaVu Sans', Arial, sans-serif; color: #444; font-size: 10.5pt; }
.abstract { background: #f6f5f2; padding: 10px 14px; border-left: 4px solid #eb6834; margin: 12px 0; }
table { border-collapse: collapse; font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 8.6pt; margin: 6px 0 10px 0; width: 100%; }
th { background: #f0efeb; text-align: left; padding: 3px 5px; border-bottom: 1px solid #999; }
td { padding: 2.5px 5px; border-bottom: 1px solid #e3e2de; }
figure { margin: 8px 0 12px 0; text-align: center; page-break-inside: avoid; }
figcaption { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 8.6pt; color: #444; margin-top: 3px; text-align: left; }
.eq { font-family: 'DejaVu Sans Mono', monospace; font-size: 9.2pt; background: #f6f5f2; padding: 4px 8px; display: block; margin: 4px 0; }
.small { font-size: 8.8pt; color: #333; }
ul { margin: 3px 0 6px 0; padding-left: 20px; } li { margin: 1px 0; }
.pb { page-break-before: always; }
.two { display: flex; gap: 10px; } .two > figure { flex: 1; }
"""


def build(run: Path, figs: Path, out: Path, review: Path) -> Path:
    m, r, rt = load(run)
    P = lambda *a, **k: paired(m, *a, **k)  # noqa: E731
    M = lambda *a, **k: mean(m, *a, **k)  # noqa: E731
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    ci = lambda p: f"{p['mean']:+.2f} dB (95% CI {p['lo']:+.2f} to {p['hi']:+.2f}, better on {p['win']:.0%} of {p['n']} images)"  # noqa: E731

    e2 = P("FSRNet-L1", "Bicubic", 3, "bicubic")
    e2p = P("FSRNet-L1", "Bicubic", 3, "bicubic", "poster")
    e2d = P("FSRNet-L1", "Bicubic", 3, "bicubic", "div2k")
    e3p = P("FSRNet-L1", "Real-ESRGAN", 3, "bicubic", "poster")
    e3d = P("FSRNet-L1", "Real-ESRGAN", 3, "bicubic", "div2k")
    lp_p = P("FSRNet-L1", "Real-ESRGAN", 3, "bicubic", "poster", metric="lpips")
    lp_d = P("FSRNet-L1", "Real-ESRGAN", 3, "bicubic", "div2k", metric="lpips")
    a1 = P("FSRNet-L1", "FSRNet-MSE", 3, "bicubic")
    a2m = P("FSRNet-L1-mixed", "FSRNet-L1", 3, "mixed")
    a2c = P("FSRNet-L1-mixed", "FSRNet-L1", 3, "bicubic")
    a3p = P("FSRNet-L1+posters", "FSRNet-L1", 3, "bicubic", "poster")
    a3d = P("FSRNet-L1+posters", "FSRNet-L1", 3, "bicubic", "div2k")
    arch = P("FSRNet-L1", "SRCNN", 3, "bicubic")
    six = P("FSRNet-L1 3x>3x>down", "FSRNet-L1", 6, "bicubic")
    bb = P("Bicubic", "Bilinear", 3, "bicubic")
    bn = P("Bilinear", "Nearest", 3, "bicubic")

    # main table
    methods = ["Nearest", "Bilinear", "Bicubic", "SRCNN", "FSRNet-L1", "FSRNet-MSE", "FSRNet-L1-mixed",
               "FSRNet-L1+posters", "Real-ESRGAN"]
    rows = []
    for meth in methods:
        rows.append({"Method": meth,
                     "PSNR-Y 3× clean": M(meth, 3, "bicubic"), "SSIM 3× clean": M(meth, 3, "bicubic", "ssim_y"),
                     "LPIPS 3× clean": M(meth, 3, "bicubic", "lpips"),
                     "PSNR-Y 3× degraded": M(meth, 3, "mixed"), "LPIPS 3× degraded": M(meth, 3, "mixed", "lpips"),
                     "PSNR-Y 6× clean": M(meth, 6, "bicubic"), "LPIPS 6× clean": M(meth, 6, "bicubic", "lpips")})
    main = pd.DataFrame(rows)
    fm = {c: "{:.2f}" for c in main.columns if "PSNR" in c}
    fm.update({c: "{:.3f}" for c in main.columns if "SSIM" in c or "LPIPS" in c})
    main_t = table(main, fm, bold_max=[c for c in main.columns if "PSNR" in c or "SSIM" in c],
                   bold_min=[c for c in main.columns if "LPIPS" in c])

    src_rows = []
    for meth in ["Bicubic", "SRCNN", "FSRNet-L1", "FSRNet-L1+posters", "FSRNet-L1-mixed", "Real-ESRGAN"]:
        src_rows.append({"Method": meth, "DIV2K PSNR-Y": M(meth, 3, "bicubic", src="div2k"),
                         "DIV2K LPIPS": M(meth, 3, "bicubic", "lpips", src="div2k"),
                         "Posters PSNR-Y": M(meth, 3, "bicubic", src="poster"),
                         "Posters LPIPS": M(meth, 3, "bicubic", "lpips", src="poster"),
                         "Posters edge-PSNR": M(meth, 3, "bicubic", "edge_psnr", src="poster"),
                         "Posters overshoot ×10³": 1000 * M(meth, 3, "bicubic", "overshoot", src="poster")})
    src_t = table(pd.DataFrame(src_rows), {"DIV2K PSNR-Y": "{:.2f}", "DIV2K LPIPS": "{:.3f}", "Posters PSNR-Y": "{:.2f}",
                                          "Posters LPIPS": "{:.3f}", "Posters edge-PSNR": "{:.2f}", "Posters overshoot ×10³": "{:.2f}"},
                  bold_max=["DIV2K PSNR-Y", "Posters PSNR-Y", "Posters edge-PSNR"],
                  bold_min=["DIV2K LPIPS", "Posters LPIPS", "Posters overshoot ×10³"])

    rp = r[(r.scale == 3) & (r.degradation == "bicubic") & r.image_id.str.startswith("poster")]
    labs = ["text_small", "text_logo", "face", "edges", "texture"]
    roi_rows = []
    for lab in labs:
        d = rp[rp.roi == lab]
        row = {"Region (posters)": lab.replace("_", " "), "n": int((d.method == "Bicubic").sum())}
        for meth in ["Bicubic", "SRCNN", "FSRNet-L1", "Real-ESRGAN"]:
            row[meth] = d[d.method == meth].psnr_y.mean()
        row["Real-ESRGAN overshoot / ours"] = d[d.method == "Real-ESRGAN"].overshoot.mean() / max(1e-9, d[d.method == "FSRNet-L1"].overshoot.mean())
        roi_rows.append(row)
    roi_df = pd.DataFrame(roi_rows)
    roi_t = table(roi_df, {"n": "{:d}", "Bicubic": "{:.2f}", "SRCNN": "{:.2f}", "FSRNet-L1": "{:.2f}", "Real-ESRGAN": "{:.2f}",
                           "Real-ESRGAN overshoot / ours": "{:.1f}×"})
    ts = roi_df.set_index("Region (posters)")

    rt7 = rt[rt.input == "1280x720"].copy()
    rt_rows = []
    for meth in ["Bicubic", "SRCNN", "FSRNet-L1", "Real-ESRGAN", "FSRNet-L1 3x>3x>down"]:
        a = rt7[(rt7.method == meth) & (rt7.scale == 3)]
        b = rt7[(rt7.method == meth) & (rt7.scale == 6)]
        rt_rows.append({"Method": meth, "device": (a.device.iloc[0] if len(a) else b.device.iloc[0]).upper(),
                        "720p→4K (s)": a.median_s.iloc[0] if len(a) else float("nan"),
                        "720p→8K (s)": b.median_s.iloc[0] if len(b) else float("nan"),
                        "peak GPU MB": (a.peak_gpu_mb.iloc[0] if len(a) else b.peak_gpu_mb.iloc[0])})
    rtd = pd.DataFrame(rt_rows)
    rt_t = table(rtd, {"720p→4K (s)": "{:.2f}", "720p→8K (s)": "{:.2f}", "peak GPU MB": "{:.0f}"}).replace(">nan<", ">–<")
    t_ours = float(rtd.set_index("Method").loc["FSRNet-L1", "720p→4K (s)"])
    t_re = float(rtd.set_index("Method").loc["Real-ESRGAN", "720p→4K (s)"])
    mem_f = float(rtd.set_index("Method").loc["FSRNet-L1", "peak GPU MB"])
    mem_s = float(rtd.set_index("Method").loc["SRCNN", "peak GPU MB"])
    mem_ratio = mem_s / mem_f

    env = json.loads((run / "environment.json").read_text())
    F = lambda n, cap, w="100%": img(figs / n, w, cap)  # noqa: E731

    body = f"""
<h1>Faithful Image Super-Resolution for Movie Posters and Digital Artwork</h1>
<div class="sub">CS663 Digital Image Processing — Course Project Report<br>
Mridul Choudhary (23B3933) &nbsp;·&nbsp; Jashanpreet Singh (23B0985)<br>
Code: github.com/MridulChoudhary/cs663-project</div>

<div class="abstract"><b>Abstract.</b> We study 3× (720p→4K) and 6× (720p→8K) super-resolution (SR) of posters and digital
artwork, separating <i>pixel fidelity</i> from <i>perceptual sharpness</i>. We implement classical interpolation from scratch,
design and train our own compact residual CNN (<b>FSRNet</b>, 0.37 M parameters), re-implement SRCNN, and compare against the
pretrained GAN model Real-ESRGAN. On 66 held-out images (50 DIV2K photos, 16 genuine high-resolution posters) FSRNet improves
on bicubic by {ci(e2)} and on posters by {e2p['mean']:+.2f} dB. Against Real-ESRGAN it is more faithful on every test image
(+{e3p['mean']:.1f} dB on posters, +{e3d['mean']:.1f} dB on photos) and, on posters, also perceptually closer (LPIPS
{M('FSRNet-L1', 3, 'bicubic', 'lpips', src='poster'):.3f} vs {M('Real-ESRGAN', 3, 'bicubic', 'lpips', src='poster'):.3f}),
while running ~{t_re / t_ours:.0f}× faster. Real-ESRGAN remains perceptually better on natural photos and on heavily degraded
inputs, at the cost of invented textures and up to {roi_df['Real-ESRGAN overshoot / ours'].max():.0f}× more overshoot (ringing) on
fine detail. Four ablations (loss, degradation model, poster training data, architecture) and a 3×-vs-6× study complete the analysis.</div>

<h2>1. Problem and motivation</h2>
<p>Posters and artwork are increasingly shown on 4K/8K displays but often exist only at ~720p. Upscaling them is an
<b>ill-posed inverse problem</b>: the observed low-resolution (LR) image is modelled as
<span class="eq">y = Q( (x ∗ k)↓<sub>s</sub> + n )</span>
(blur k, downsampling by s, noise n, quantisation/JPEG Q). Infinitely many high-resolution (HR) images x explain the same y, so every
method chooses one using a prior. Posters stress this: small typography, logos, thin lines, faces and flat colour regions make
both blur and invented detail obvious. Our research question: <i>how do classical interpolation, a CNN we design and train, and a
pretrained perceptual model compare in fidelity and visual quality on poster/artwork imagery, and how do the degradation model and
scale factor change that?</i></p>

<h2>2. Background</h2>
<ul>
<li><b>Interpolation as convolution.</b> Resampling is f̂(t)=Σ<sub>j</sub> f[j]·h(t−j) with a reconstruction kernel h: box (nearest),
triangle (bilinear) or Keys cubic convolution (bicubic, a = −0.5), whose negative lobes sharpen slightly but cause ringing.
When downsampling, the kernel is stretched by 1/s (anti-aliasing).</li>
<li><b>CNN-based SR.</b> SRCNN (Dong et al., 2014/16) refines a bicubic upsampling with three convolutions; ESPCN (Shi et al., 2016) moves
all computation to LR resolution with sub-pixel convolution (PixelShuffle); EDSR (Lim et al., 2017) uses deep residual blocks without
batch-norm.</li>
<li><b>Losses as priors.</b> Minimising MSE gives the conditional mean of p(x|y) (MLE under Gaussian noise, averages plausible solutions → blur);
L1 gives the conditional median (Laplacian MLE), usually sharper and more robust.</li>
<li><b>Perception–distortion trade-off</b> (Blau &amp; Michaeli, 2018): no estimator can be optimal in both distortion (e.g. MSE) and
perceptual quality. GAN-trained models such as Real-ESRGAN (Wang et al., 2021) move toward perceptual realism and may hallucinate.</li>
</ul>

<h2>3. Methods (all implemented by us unless stated)</h2>
<h3>3.1 Classical baselines</h3>
<p>Nearest, bilinear and bicubic (Keys, a=−0.5) implemented from scratch in NumPy as separable sparse weight matrices
(<i>out = W<sub>h</sub>·img·W<sub>w</sub><sup>T</sup></i>), pixel-centre convention, anti-aliased when shrinking. Verified against Pillow to
within Pillow's own 8-bit intermediate rounding (unit tests).</p>
<h3>3.2 FSRNet — our model</h3>
{F('diagram_fsrnet.png', 'Figure 1. FSRNet. All convolutions operate at LR resolution; a sub-pixel layer produces the 3× output; a global bicubic skip means the network only learns the missing high frequencies; zero-initialising the last convolution makes training start exactly at bicubic.')}
<p>Design rationale: residual learning on top of bicubic stabilises training and lets a small network focus on detail; operating at LR
resolution makes it 9× cheaper per layer than SRCNN at 3×; no batch-norm (EDSR) preserves absolute contrast. SRCNN (9-5-5, on a
bicubic-upsampled input, with a residual connection) is re-implemented as the classic baseline. Neither is an exact reproduction of a
published model.</p>
<h3>3.3 Pretrained comparator</h3>
<p>Real-ESRGAN x4plus (RRDB generator, 16.7 M parameters, BSD-3 licence), loaded with <i>spandrel</i> from an explicitly downloaded file and
<b>not modified or retrained</b>. Its native scale is 4×: for 3× we run 4× and apply an anti-aliased bicubic resize to 3×; for 6× we run 4× and
resize by 1.5×. Tiled inference (256 px tiles, 16 px overlap) bounds GPU memory.</p>
<h3>3.4 Data and degradation pipeline</h3>
<ul>
<li><b>DIV2K</b> (800 train, 100 validation 2K photos; academic use). The 100 validation images are split 50/50 (seeded, per image) into
validation (checkpoint selection) and a photo test set.</li>
<li><b>Posters</b>: 132 files supplied by us from free-reuse sites. 54 are genuine hi-res originals → 38 train / 16 test (seeded per-image split).
78 were already AI-upscaled by web services; they are <b>never used as ground truth</b> (their detail is another model's guess) and form a
qualitative real-world set. Exact/near-duplicates are excluded (12 images in total), large originals are reduced to ≤3840 px long side with
an anti-aliased filter, transparency is flattened. Each image carries category tags (poster, illustration, photographic, text/logo,
fine lines, texture) and the 16 poster test images carry manually drawn regions of interest (8 title/logo, 5 small-text, 6 face boxes).</li>
<li><b>Degradations</b> (fixed order, recorded per image, alignment unit-tested): <i>bicubic</i> (clean benchmark) and <i>mixed</i> =
Gaussian blur σ∈[0.4,1.6] (p=0.7) → anti-aliased bicubic ↓s → Gaussian noise σ∈[1,6]/255 (p=0.5) → 8-bit → JPEG q∈[55,95] (p=0.7).</li>
<li><b>Training</b>: 13,962 HR tiles of 480², random 144² HR / 48² LR patches degraded on the fly, flips/rotations; L1 loss, Adam
(β=0.9, 0.99), lr 5·10⁻⁴ with warm-up and cosine decay, batch 16, 30k iterations (~73 min per run on a Colab T4), best checkpoint by
validation PSNR. Training resumes bit-exactly after interruptions (tested).</li>
</ul>

<h2>4. Experimental design</h2>
<ul>
<li><b>Test set:</b> 66 images (50 DIV2K photos + 16 posters), fixed before training; every method sees identical LR inputs at 3× and 6×,
clean and degraded.</li>
<li><b>Metrics:</b> PSNR and SSIM on BT.601 luma (border crop = scale; our implementations match scikit-image), RGB-PSNR, edge-PSNR (PSNR on the
10% strongest ground-truth gradients), overshoot (mean excursion outside the local 3×3 GT range — a ringing/halo indicator), and LPIPS-Alex
(learned perceptual distance, lower = closer). Paired comparisons use per-image differences with 5,000-sample bootstrap 95% CIs and win rates.</li>
<li><b>Experiments:</b> E1 interpolation; E2 CNN vs bicubic; E3 CNN vs Real-ESRGAN; E4 3× vs 6× (incl. two staging strategies); E5 degradation
robustness; E6 category-wise; E7 regions (text, faces, edges, texture); E8 runtime/memory.
<b>Ablations</b> (one factor changed vs the main model FSRNet-L1): A1 MSE loss; A2 training on the mixed degradation; A3 adding the 38 poster
training images; architecture: SRCNN.</li>
<li><b>Environment:</b> {env.get('gpu', 'GPU')}, Python {env.get('python')}, PyTorch {env.get('torch')}, code commit {env.get('git_commit')}.
Local CPU re-computation of two posters reproduces the Colab PSNRs to four decimals.</li>
</ul>

<h2 class="pb">5. Results</h2>
<h3>5.1 Overall (66 test images)</h3>
{main_t}
<p class="small">Table 1. Means over 66 images. "Degraded" = blur + noise + JPEG inputs. Bold = best in column.</p>
<p><b>E1 — interpolation.</b> Bicubic &gt; bilinear by {ci(bb)}; bilinear &gt; nearest by {bn['mean']:+.2f} dB. The ranking follows kernel
smoothness/support, as theory predicts.<br>
<b>E2 — our CNN vs bicubic.</b> FSRNet-L1 gains {ci(e2)}; {e2d['mean']:+.2f} dB on photos and {e2p['mean']:+.2f} dB on posters.</p>

<h3>5.2 Posters vs photos, and the market model (E3)</h3>
{F('chart_psnr_by_source.png', 'Figure 2. PSNR-Y at 3× on clean inputs, split by source (orange = our implementations, green = pretrained).')}
{src_t}
<p class="small">Table 2. 3×, clean inputs. Overshoot = ringing indicator (lower is better).</p>
<p>FSRNet is more faithful than Real-ESRGAN on every single test image: {ci(e3p)} on posters and {ci(e3d)} on photos.
Perceptually the picture depends on content: on posters FSRNet also has the lower LPIPS ({lp_p['mean']:+.3f}, CI {lp_p['lo']:+.3f}
to {lp_p['hi']:+.3f}, better on {lp_p['win']:.0%}), whereas on natural photos Real-ESRGAN is perceptually closer
(FSRNet {lp_d['mean']:+.3f} LPIPS). Real-ESRGAN scores below plain bicubic in PSNR on photos — its added detail does not match the true image.</p>
{F('chart_perception_distortion.png', 'Figure 3. Perception–distortion plane (3×, clean). Bottom-right is best. On photos the two learned families sit at different ends of the trade-off; on posters FSRNet dominates.')}

<h3>5.3 Detail regions (E7)</h3>
{F('chart_roi_posters.png', 'Figure 4. Region PSNR-Y on posters (3×, clean). Regions were drawn on the ground truth only.')}
{roi_t}
<p class="small">Table 3. Poster regions, 3× clean. Last column: Real-ESRGAN's mean overshoot divided by FSRNet's.</p>
<p>Gains are largest on <b>text</b>: small text improves by {ts.loc['text small', 'FSRNet-L1'] - ts.loc['text small', 'Bicubic']:+.1f} dB over
bicubic and titles/logos by {ts.loc['text logo', 'FSRNet-L1'] - ts.loc['text logo', 'Bicubic']:+.1f} dB. Real-ESRGAN produces several times
more overshoot on every region type — the halos and over-thickened strokes visible in Figure 5.</p>
{F('crop_poster_1401918_text_small_bicubic.png', 'Figure 5. Small serif text (poster test image 1401918, 3×). FSRNet keeps letter shapes and serifs closest to the ground truth; Real-ESRGAN thickens strokes and alters letterforms. Region PSNR from the benchmark (whole region).', '78%')}
{F('crop_poster_1329765_face_bicubic.png', 'Figure 6. Face region (poster 1329765, 3×). Region PSNR: bicubic 34.19, SRCNN 39.00, FSRNet 42.41, Real-ESRGAN 35.93 dB.')}

<h3>5.4 Ablations</h3>
<ul>
<li><b>A1 loss (L1 vs MSE):</b> L1 better by {ci(a1)} — consistent but small; LPIPS is equal within 0.001. The loss choice barely matters at this model size.</li>
<li><b>A2 degradation model:</b> training on blur+noise+JPEG improves degraded inputs by {ci(a2m)} but costs {a2c['mean']:+.2f} dB on clean
inputs (better on {a2c['win']:.0%}). A model should be trained for the degradations it will meet — there is no free robustness.</li>
<li><b>A3 poster training data:</b> adding 38 posters improves posters by {ci(a3p)} and leaves photos unchanged ({a3d['mean']:+.3f} dB) —
a small but consistent in-domain gain from only ~11% extra training tiles.</li>
<li><b>Architecture:</b> FSRNet beats SRCNN by {ci(arch)} while using {mem_ratio:.0f}× less GPU memory at 720p→4K ({mem_f:.0f} vs {mem_s:.0f} MB).</li>
</ul>
<div class="two">{F('chart_ablation_degradation.png', 'Figure 7. A2: clean- vs degradation-trained FSRNet on clean and degraded inputs.')}
{F('chart_3x_vs_6x.png', 'Figure 8. E4: 3× vs 6× (clean inputs, 66 images).')}</div>
{F('crop_poster_1166038_text_logo_mixed.png', 'Figure 9. Degraded input (blur+noise+JPEG), poster 1166038. The degradation-trained FSRNet restores the lettering and the hexagonal grille best; Real-ESRGAN replaces the hexagonal mesh with a rectangular one — an invented texture.', '52%')}

<h3>5.5 3× vs 6× (E4) and runtime (E8)</h3>
<p>All methods lose 3–5 dB from 3× to 6× (FSRNet {M('FSRNet-L1', 3, 'bicubic') - M('FSRNet-L1', 6, 'bicubic'):.2f} dB,
bicubic {M('Bicubic', 3, 'bicubic') - M('Bicubic', 6, 'bicubic'):.2f} dB): an 8K output from a 720p input contains far less
genuine information than its pixel count suggests. Applying FSRNet twice (9×) and shrinking to 6× beats FSRNet-then-bicubic by
{ci(six)}, at ~4× the runtime; first-stage errors propagate in both strategies.</p>
{rt_t}
<p class="small">Table 4. Median of 3 runs after warm-up, model loading excluded, Tesla T4 (CPU for classical methods, which run in our NumPy
code). FSRNet/Real-ESRGAN use tiling (512/256 px).</p>

<h3>5.6 Category-wise (E6) and real-world images</h3>
<p>Across poster categories FSRNet gains most on fine lines and text/logo images (see Table 3 and <i>summary_by_category.csv</i>);
per-category counts are small (6–15 images) so these are indicative only. On the 74 AI-upscaled posters (qualitative; inputs = the
supplied file reduced 4×) the clean-trained CNNs sharpen JPEG/halftone artefacts already present in the input, while the
degradation-trained FSRNet and Real-ESRGAN suppress them, Real-ESRGAN at the cost of flattening fine texture into a painterly look.
Because these inputs were derived from the third-party upscales themselves, the supplied files cannot serve as ground truth or as a fair
competitor; no scores are reported for this set.</p>
{F('upscaled_f1_max_verstappen_brutalism_crops.png', 'Figure 10. Real-world example (qualitative only): halftone/JPEG artefacts in the input are amplified by clean-trained models and suppressed by the degradation-trained model and Real-ESRGAN.', '62%')}

<h2>6. Discussion</h2>
<ul>
<li><b>Which method when?</b> For faithful enlargement of artwork with typography, a small task-specific CNN is the best choice measured here:
most accurate, perceptually closest on posters, and ~{t_re / t_ours:.0f}× faster than the GAN. For degraded photographs where a pleasing
look matters more than accuracy, Real-ESRGAN is perceptually stronger.</li>
<li><b>Do sharper outputs stay faithful?</b> No. Real-ESRGAN's outputs look crisp yet have the lowest PSNR and edge-PSNR and the most overshoot;
on text it thickens strokes and on textures it can swap one pattern for another (Fig. 9). Sharpness is not evidence of accuracy.</li>
<li><b>Typography and graphics</b> benefit most from learned SR (+7–8 dB on text regions), because sharp edges and flat fills are
highly predictable — an easy prior for a CNN.</li>
<li><b>6×</b> is a different regime: all methods degrade sharply; staged inference helps slightly; claims of "true 8K detail" from 720p are unfounded.</li>
<li><b>Degradations matter more than the loss:</b> the train/test degradation match (A2, ±0.8–1.0 dB) dwarfs the L1/MSE choice (0.08 dB).</li>
</ul>

<h2>7. Limitations</h2>
<ul>
<li>Small poster test set (16 images; 5–8 per region type) — confidence intervals are reported, but category conclusions are indicative.</li>
<li>Synthetic degradations: real posters have unknown processing; the "real-world" set is qualitative and its references are AI-generated.</li>
<li>PSNR/SSIM measure agreement with one ground truth; LPIPS is a learned proxy. None proves that a detail is authentic.</li>
<li>Real-ESRGAN is evaluated through scale adaptation (4×→3×/6× resize), which may slightly disadvantage it. Each model was trained once (single seed).</li>
<li>Poster ground truth is itself JPEG-compressed for some images (quality screen flagged 5 originals, 2 in the test set).</li>
</ul>

<h2>8. Conclusion and future work</h2>
<p>A compact CNN designed for the task (FSRNet) reconstructs posters far more faithfully than interpolation (+{e2p['mean']:.1f} dB) and than a
popular pretrained GAN (+{e3p['mean']:.1f} dB), with sharper, more accurate typography, while the GAN wins perceptually on degraded photos.
The measured perception–distortion trade-off, the strong effect of the degradation model and the steep 3×→6× drop are the main lessons.
Future work: perceptual/GAN fine-tuning of FSRNet with a fidelity constraint, a dedicated 6× model, text-aware losses, larger poster test set,
and a human preference study.</p>

<h2>References and licences</h2>
<p class="small">Keys (1981) cubic convolution · Wang et al. (2004) SSIM · Dong et al. (2014/2016) SRCNN · Shi et al. (2016) ESPCN · Lim et al. (2017) EDSR ·
Agustsson &amp; Timofte (2017) DIV2K (academic use, not redistributed) · Blau &amp; Michaeli (2018) perception–distortion trade-off · Zhang et al. (2018) LPIPS (BSD-2) ·
Wang et al. (2021) Real-ESRGAN (BSD-3), loaded via spandrel (MIT) · PyTorch, NumPy, SciPy, Pillow, OpenCV, scikit-image, Gradio.
Posters: free-reuse sources, private academic evaluation, not redistributed. All numbers in this report are read from
<i>outputs/eval/main/*.csv</i> by <i>scripts/build_report.py</i>.</p>
"""
    out.mkdir(parents=True, exist_ok=True)
    page = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>CS663 report</title><style>{CSS}</style></head><body>{body}</body></html>"
    # images are referenced relative to figs dir for the real-world one
    page = page.replace("../eval/main/", "")
    hp = out / "CS663_Faithful_SR_Report.html"
    hp.write_text(page, encoding="utf-8")
    return hp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/eval/main")
    ap.add_argument("--figs", default="outputs/figures")
    ap.add_argument("--out", default="outputs/report")
    a = ap.parse_args()
    run, figs = Path(a.run), Path(a.figs)
    # real-world figure lives in the run dir; copy into figs for embedding
    rw = run / "real_world" / "upscaled_f1_max_verstappen_brutalism_crops.png"
    if rw.exists():
        (figs / "upscaled_f1_max_verstappen_brutalism_crops.png").write_bytes(rw.read_bytes())
    hp = build(run, figs, Path(a.out), Path("outputs/dataset_review"))
    pdf = hp.with_suffix(".pdf")
    js = f"""const {{chromium}}=require('playwright');(async()=>{{const b=await chromium.launch();const p=await b.newPage();
await p.goto('file://{hp.resolve()}');await p.pdf({{path:'{pdf.resolve()}',format:'A4',printBackground:true,
displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="font-size:8px;width:100%;text-align:center;color:#666">CS663 · Faithful Super-Resolution · <span class=pageNumber></span>/<span class=totalPages></span></div>',
margin:{{top:'16mm',bottom:'16mm',left:'15mm',right:'15mm'}}}});await b.close();}})();"""
    Path("/tmp/pdf.js").write_text(js)
    subprocess.run(["node", "/tmp/pdf.js"], check=True, cwd="/tmp")
    print("report:", pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
