// Build the 10-minute CS663 presentation (.pptx) from saved results.
// node scripts/build_deck.js   (run from the project root)
// Numbers come from outputs/report/deck_numbers.json (written from the benchmark CSVs).
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");
// Theme colours are written by a helper from Claude's pptx skill; without it the deck uses Office's stock palette.
const applyTheme = process.env.PPTX_SKILL ? require(process.env.PPTX_SKILL + "/scripts/apply_theme.js").applyTheme
  : async () => console.warn("PPTX_SKILL not set: theme colours not applied");

const N = JSON.parse(fs.readFileSync("outputs/report/deck_numbers.json"));
const FIG = "outputs/figures/";
const OUT = "outputs/report/CS663_Faithful_SR_Presentation.pptx";

const THEME = {
  name: "Faithful SR",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1B1F", lt1: "FFFFFF", dk2: "23263A", lt2: "F3F3F1",
    accent1: "EB6834", accent2: "2A78D6", accent3: "1BAF7A", accent4: "F5B48F",
    accent5: "6B6A66", accent6: "9EC3EF", hlink: "2A78D6", folHlink: "6B6A66",
  },
};
const HEX = { ours: "EB6834", classical: "2A78D6", pre: "1BAF7A", srcnn: "F5B48F", grid: "E4E3DF", muted: "6B6A66", ink: "1B1B1F", dark: "23263A" };
const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2);
const sgn = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
  pres.title = "Faithful Image Super-Resolution for Movie Posters and Digital Artwork";
  pres.author = "Mridul Choudhary, Jashanpreet Singh";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  // ---------- layouts
  pres.defineSlideMaster({
    title: "TITLE", background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.2, w: 7.0, h: 2.35, fontSize: 36, bold: true, color: C.text2, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 },
    margin: [0.5, 0.6, 0.5, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 30, bold: true, color: C.text2, valign: "middle" }, text: "" } },
      { text: { text: "CS663 · Faithful Super-Resolution", options: { x: 0.6, y: 7.05, w: 6, h: 0.3, fontSize: 10, color: HEX.muted } } },
    ],
    slideNumber: { x: 12.2, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: HEX.muted, align: "right" },
  });

  // ---------- helpers
  async function pic(slide, file, x, y, w, h, align = "center") {
    const meta = await sharp(file).metadata();
    const r = Math.min(w / meta.width, h / meta.height);
    const iw = meta.width * r, ih = meta.height * r;
    const ix = align === "left" ? x : x + (w - iw) / 2;
    slide.addImage({ path: file, x: ix, y: y + (h - ih) / 2, w: iw, h: ih, objectName: path.basename(file) });
    return { x: ix, y: y + (h - ih) / 2, w: iw, h: ih };
  }
  function txt(slide, text, o) { slide.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top" }, o)); }
  function stat(slide, big, small, x, y, w, col) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.55, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: "stat card" });
    txt(slide, big, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.75, fontSize: 36, bold: true, color: col, fontFace: "Arial" });
    txt(slide, small, { x: x + 0.25, y: y + 0.9, w: w - 0.5, h: 0.6, fontSize: 13, color: HEX.muted });
  }
  function bullets(slide, items, o) {
    slide.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
      Object.assign({ isTextBox: true, fontSize: 16, color: C.text1, paraSpaceAfter: 8, valign: "top", margin: 0.05 }, o));
  }
  const chartBase = (title) => ({
    showTitle: !!title, title, titleFontSize: 13, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
    dataLabelFontSize: 10, dataLabelColor: HEX.ink, dataLabelFontFace: "+mn-lt",
  });

  // ================= 1. Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
  txt(s, "CS663 · DIGITAL IMAGE PROCESSING · COURSE PROJECT", { x: 0.6, y: 0.75, w: 6.9, h: 0.35, fontSize: 13, bold: true, color: C.accent1, charSpacing: 1 });
  s.addText("Faithful Image Super-Resolution for Movie Posters and Digital Artwork", { placeholder: "title" });
  txt(s, "720p → 4K / 8K: how much detail can we really recover — and who invents it?", { x: 0.6, y: 3.75, w: 6.9, h: 0.8, fontSize: 18, italic: true, color: HEX.muted });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.85, w: 6.9, h: 1.25, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: "authors card" });
  [["Mridul Choudhary", "23B3933"], ["Jashanpreet Singh", "23B0985"]].forEach(([nm, roll], k) => {
    const x = 0.9 + k * 3.35;
    s.addShape(pres.shapes.OVAL, { x, y: 5.17, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "author dot " + k });
    txt(s, nm.split(" ").map((w) => w[0]).join(""), { x, y: 5.17, w: 0.6, h: 0.6, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, nm, { x: x + 0.75, y: 5.12, w: 2.5, h: 0.4, fontSize: 17, bold: true, color: HEX.dark });
    txt(s, roll, { x: x + 0.75, y: 5.5, w: 2.5, h: 0.35, fontSize: 14, color: HEX.muted });
  });
  const famLeg = [[HEX.classical, "Classical"], [HEX.ours, "FSRNet (ours)"], [HEX.pre, "Real-ESRGAN"]];
  famLeg.forEach(([col, lab], k) => {
    const x = 0.6 + k * 2.3;
    s.addShape(pres.shapes.OVAL, { x, y: 6.5, w: 0.22, h: 0.22, fill: { color: col }, line: { color: col }, objectName: "legend " + k });
    txt(s, lab, { x: x + 0.32, y: 6.45, w: 1.9, h: 0.32, fontSize: 13, color: HEX.muted, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.95, y: 0.55, w: 4.8, h: 6.4, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.15, objectName: "hero card" });
  await pic(s, FIG + "slide_poster_1401918_text_small_bicubic.png", 8.15, 0.75, 4.4, 6.0);
  s.addNotes("[0:00–0:30] Introduce ourselves and the question: when we enlarge a poster from 720p to 4K or 8K, which methods recover the real detail, and which ones make it up? We built and trained our own network and compared it with interpolation and with a popular pretrained upscaler.");

  // ================= 2. Problem
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Super-resolution is an ill-posed inverse problem", { placeholder: "title" });
  await pic(s, FIG + "slide_problem_poster.png", 0.6, 1.45, 5.6, 3.3);
  txt(s, "A 3840×2160 test poster. Orange box: region shown on the right.", { x: 0.6, y: 4.85, w: 5.6, h: 0.4, fontSize: 12, color: HEX.muted });
  txt(s, "What a 720p copy contains", { x: 6.7, y: 1.45, w: 6, h: 0.35, fontSize: 14, bold: true, color: HEX.muted });
  await pic(s, FIG + "slide_problem_lr.png", 6.7, 1.85, 6.0, 1.35, "left");
  txt(s, "What the 4K poster really looks like", { x: 6.7, y: 3.35, w: 6, h: 0.35, fontSize: 14, bold: true, color: HEX.muted });
  await pic(s, FIG + "slide_problem_hr.png", 6.7, 3.75, 6.0, 1.35, "left");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.45, w: 12.1, h: 1.35, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: "model box" });
  s.addText([
    { text: "y = Q( (x ∗ k)↓s + n )", options: { bold: true, fontFace: "Cambria", fontSize: 20, color: HEX.dark, breakLine: true } },
    { text: "blur k, downsample by s, noise n, 8-bit/JPEG Q: infinitely many sharp images x explain the same y — every method picks one using a prior. Posters make mistakes visible: small type, logos, thin lines, faces, flat colour.", options: { fontSize: 15, color: C.text1 } },
  ], { x: 0.85, y: 5.55, w: 11.6, h: 1.15, isTextBox: true, margin: 0, valign: "middle" });
  s.addNotes("[0:30–1:30] The degradation model: blur, downsampling, noise and compression. Information is lost, so any upscaler must guess. Posters are especially unforgiving: typography and thin graphic lines show blur and invented detail immediately. Our research question: how do interpolation, a CNN we train, and a pretrained perceptual GAN compare on fidelity versus visual quality, and how do the degradation model and the scale factor change that?");

  // ================= 3. What we did
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("What we did — mapped to the project expectations", { placeholder: "title" });
  const steps = [
    ["1", "Papers", "SRCNN, ESPCN, EDSR, Real-ESRGAN, perception–distortion trade-off"],
    ["2", "Maths", "Interpolation kernels, inverse problem, L1/L2 as Laplacian/Gaussian MLE, PSNR/SSIM"],
    ["3", "From scratch", "NumPy interpolation, degradation pipeline, metrics, FSRNet + training loop, tiling"],
    ["4", "Data", "DIV2K + 54 genuine hi-res posters (tagged, regions drawn) + 74 AI-upscaled posters"],
    ["5", "Experiments", `8 experiments + 4 ablations · ${N.n_results.toLocaleString("en-US")} measured results · bootstrap CIs`],
    ["6", "Results", "Fidelity vs perception, text/faces, robustness, 3× vs 6×, runtime"],
  ];
  steps.forEach(([n, h, d], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.5 + row * 2.6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.85, h: 2.3, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: "item " + n });
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "num " + n });
    txt(s, n, { x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, h, { x: x + 1.0, y: y + 0.3, w: 2.7, h: 0.5, fontSize: 20, bold: true, color: HEX.dark, valign: "middle" });
    txt(s, d, { x: x + 0.25, y: y + 1.05, w: 3.4, h: 1.15, fontSize: 14, color: C.text1 });
  });
  s.addNotes("[1:30–2:00] Quick roadmap, structured by the six things the course asks for. Everything except the pretrained comparator was implemented by us, including the interpolation kernels and the metrics, which we unit-tested against Pillow and scikit-image.");

  // ================= 4. Methods
  pres.addSection({ title: "Method" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Three families of upscalers, compared on identical inputs", { placeholder: "title" });
  await pic(s, FIG + "diagram_fsrnet.png", 0.6, 1.35, 12.1, 3.55);
  const fam = [
    [HEX.classical, "Classical (ours, NumPy)", "Nearest · bilinear · bicubic (Keys a = −0.5) as convolution with a reconstruction kernel; sparse separable matrices"],
    [HEX.ours, "Learned (ours, PyTorch)", "FSRNet: 8 residual blocks at LR resolution + sub-pixel ×3 + bicubic skip, 0.37 M params. SRCNN re-implemented as baseline"],
    [HEX.pre, "Pretrained (market)", "Real-ESRGAN x4plus: GAN-trained RRDB, 16.7 M params, used as-is (4× output resized to 3×/6×)"],
  ];
  fam.forEach(([col, h, d], i) => {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.OVAL, { x, y: 5.15, w: 0.32, h: 0.32, fill: { color: col }, line: { color: col }, objectName: "dot " + i });
    txt(s, h, { x: x + 0.45, y: 5.1, w: 3.5, h: 0.4, fontSize: 16, bold: true, color: HEX.dark });
    txt(s, d, { x, y: 5.6, w: 3.85, h: 1.3, fontSize: 13, color: C.text1 });
  });
  s.addNotes("[2:00–3:10] Classical interpolation is convolution with a kernel; bicubic's negative lobes sharpen but ring. FSRNet is our network: all convolutions run at low resolution, a sub-pixel layer produces the 3x image, and a bicubic skip means it only learns the missing high frequencies; the last layer starts at zero, so training starts from bicubic. Loss is L1, the conditional median. Real-ESRGAN is the market comparator: a much larger GAN-trained model that we did not modify.");

  // ================= 5. Data & degradation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Data, degradations and a strict test protocol", { placeholder: "title" });
  await pic(s, "outputs/dataset_review/review_test_rois.png", 6.9, 1.35, 5.8, 5.5);
  bullets(s, [
    "Train: 800 DIV2K photos (13,962 tiles); +38 posters only in ablation A3",
    "Test, fixed before training: 50 DIV2K photos + 16 genuine hi-res posters",
    "78 AI-upscaled posters: never ground truth → qualitative real-world set",
    "Split per source image; duplicates removed; tags + hand-drawn text/face regions",
    "Degradations: clean bicubic ↓, and 'mixed' = blur → ↓ → noise → JPEG",
    "3× (→4K) and 6× (→8K); 30k iterations per model, ~73 min on a Colab T4",
  ], { x: 0.6, y: 1.45, w: 6.0, h: 5.3, fontSize: 16 });
  s.addNotes("[3:10–4:00] Data: DIV2K for training plus our posters. We separated genuine high-resolution originals from posters that had already been AI-upscaled by web tools — those can never be ground truth. Splits are by source image, so no crop of a test poster is ever seen in training. Right: the 16 test posters with the text, face and edge regions we evaluate separately.");

  // ================= 6. Metrics
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Measuring fidelity and perception separately", { placeholder: "title" });
  const mets = [
    ["PSNR / SSIM (luma)", "Pixel and structural agreement with the true image. Our code matches scikit-image exactly."],
    ["Edge-PSNR · overshoot", "Fidelity on the strongest edges (text, lines); overshoot = ringing/halos outside the true local range."],
    ["LPIPS", "Learned perceptual distance — how similar it looks to a deep network. A proxy, not proof of authenticity."],
    ["Paired statistics", "Per-image differences vs a reference, 95% bootstrap confidence intervals and win rates."],
  ];
  mets.forEach(([h, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.15, y = 1.5 + row * 2.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 1.75, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: "metric " + i });
    txt(s, h, { x: x + 0.3, y: y + 0.2, w: 5.4, h: 0.45, fontSize: 19, bold: true, color: HEX.dark });
    txt(s, d, { x: x + 0.3, y: y + 0.75, w: 5.4, h: 0.9, fontSize: 14, color: C.text1 });
  });
  stat(s, N.n_results.toLocaleString("en-US"), "results: 66 images × 2 scales × 2 input types × 9–10 methods", 0.6, 5.5, 6.0, C.accent1);
  stat(s, "4 decimals", "local CPU re-run reproduces the Colab GPU PSNRs exactly", 6.75, 5.5, 5.95, C.accent2);
  s.addNotes("[4:00–4:45] We deliberately measure fidelity and perceptual quality separately, because the literature shows they trade off. PSNR and SSIM measure agreement with the real image; LPIPS measures perceptual similarity. All comparisons are paired per image with bootstrap confidence intervals.");

  // ================= 7. Main result
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Our network beats interpolation on every test image", { placeholder: "title" });
  const order = ["Nearest", "Bilinear", "Bicubic", "SRCNN", "FSRNet-L1", "Real-ESRGAN"];
  const labels = ["Nearest", "Bilinear", "Bicubic", "SRCNN", "FSRNet (ours)", "Real-ESRGAN"];
  const colsBar = [HEX.classical, HEX.classical, HEX.classical, HEX.srcnn, HEX.ours, HEX.pre];
  [["poster", "Posters (16) — PSNR-Y, dB", 0.6], ["div2k", "DIV2K photos (50) — PSNR-Y, dB", 4.75]].forEach(([src, title, x]) => {
    const vals = order.map((m) => +N.bysrc[src][m].toFixed(1));
    s.addChart(pres.charts.BAR, [{ name: title, labels, values: vals }], Object.assign(chartBase(title), {
      x, y: 1.35, w: 4.0, h: 4.1, barDir: "col", chartColors: colsBar, showLegend: false, showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: src === "poster" ? 28 : 26,
      valAxisMaxVal: src === "poster" ? 41 : 33, catAxisLabelFontSize: 9, valAxisHidden: true, barGapWidthPct: 45,
    }));
  });
  stat(s, sgn(N.e2.mean) + " dB", `vs bicubic, all 66 images (95% CI ${sgn(N.e2.lo)} to ${sgn(N.e2.hi)}); better on ${Math.round(N.e2.win * 100)}%`, 9.0, 1.45, 3.7, C.accent1);
  stat(s, sgn(N.e2p.mean, 1) + " dB", "on posters — flat colour and sharp edges are easy to learn", 9.0, 3.15, 3.7, C.accent1);
  stat(s, sgn(N.arch.mean) + " dB", "FSRNet vs SRCNN (same data, loss, schedule)", 9.0, 4.85, 3.7, C.accent2);
  txt(s, "3×, clean bicubic inputs. Orange = our implementations; green = pretrained.", { x: 0.6, y: 5.6, w: 8.2, h: 0.35, fontSize: 12, color: HEX.muted });
  s.addNotes(`[4:45–5:45] Main result. On all 66 test images our network beats bicubic, by ${f2(N.e2.mean)} dB on average, with a tight confidence interval, and on posters by ${f1(N.e2p.mean)} dB. It also beats our SRCNN re-implementation by about ${f1(N.arch.mean)} dB with identical training. Note where Real-ESRGAN sits: near or below bicubic in PSNR.`);

  // ================= 8. Perception-distortion
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Sharper is not the same as more accurate", { placeholder: "title" });
  await pic(s, FIG + "chart_perception_distortion.png", 0.6, 1.3, 8.2, 3.3);
  stat(s, "+" + f1(N.e3p.mean) + " dB", "FSRNet vs Real-ESRGAN on posters — better on 100% of images", 9.1, 1.4, 3.6, C.accent1);
  stat(s, "+" + f1(N.e3d.mean) + " dB", "on photos — Real-ESRGAN falls below plain bicubic in PSNR", 9.1, 3.1, 3.6, C.accent1);
  bullets(s, [
    `Posters: ours is also perceptually closer (LPIPS ${N.lpips_src.poster["FSRNet-L1"].toFixed(3)} vs ${N.lpips_src.poster["Real-ESRGAN"].toFixed(3)})`,
    `Photos: Real-ESRGAN looks closer (LPIPS ${N.lpips_src.div2k["Real-ESRGAN"].toFixed(3)} vs ${N.lpips_src.div2k["FSRNet-L1"].toFixed(3)}) — the perception–distortion trade-off`,
    "Its convincing detail is partly invented — next slide",
  ], { x: 0.6, y: 4.95, w: 12.1, h: 1.9, fontSize: 16 });
  s.addNotes("[5:45–6:30] The perception-distortion plane: right is more faithful, down is more perceptually similar. On photos the two learned approaches sit at opposite ends of the trade-off Blau and Michaeli proved. On posters our network wins on both axes, because poster content is predictable. Real-ESRGAN looks sharp but its pixels disagree with the real image.");

  // ================= 9. Text & faces
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Biggest gains on typography — where errors show most", { placeholder: "title" });
  await pic(s, FIG + "slide_poster_1401918_text_small_bicubic.png", 0.6, 1.3, 6.9, 5.6, "left");
  const rl = ["text_small", "text_logo", "face", "edges", "texture"];
  const rlab = ["small text", "title/logo", "faces", "edges", "texture"];
  s.addChart(pres.charts.BAR, ["Bicubic", "FSRNet-L1", "Real-ESRGAN"].map((m, i) => ({
    name: ["Bicubic", "FSRNet (ours)", "Real-ESRGAN"][i], labels: rlab, values: rl.map((r) => +N.roi[r][m].toFixed(1)),
  })), Object.assign(chartBase("Region PSNR-Y on posters (dB)"), {
    x: 7.8, y: 1.3, w: 4.95, h: 3.6, barDir: "col", chartColors: [HEX.classical, HEX.ours, HEX.pre],
    showLegend: true, legendPos: "b", legendFontSize: 10, legendFontFace: "+mn-lt", showValue: false, valAxisMinVal: 20, valAxisMaxVal: 40, barGapWidthPct: 60,
  }));
  bullets(s, [
    `Small text: ${sgn(N.roi.text_small["FSRNet-L1"] - N.roi.text_small["Bicubic"], 1)} dB vs bicubic`,
    `Real-ESRGAN: ${f1(Math.min(...rl.map(r => N.overshoot_ratio[r])))}–${f1(Math.max(...rl.map(r => N.overshoot_ratio[r])))}× more overshoot (halos, thick strokes)`,
  ], { x: 7.8, y: 5.1, w: 4.95, h: 1.7, fontSize: 15 });
  s.addNotes("[6:30–7:15] We drew regions on the ground truth for titles, small text and faces. The gains are largest on text. In the crop: bicubic is soft; our network keeps the serifs; Real-ESRGAN makes strokes heavier and changes letter shapes — and measures several times more overshoot on every region type.");

  // ================= 10. Ablations
  pres.addSection({ title: "Analysis" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Analysis" });
  s.addText("Ablations: degradations matter far more than the loss", { placeholder: "title" });
  const am = ["Bicubic", "FSRNet-L1", "FSRNet-L1-mixed", "Real-ESRGAN"];
  s.addChart(pres.charts.BAR, [
    { name: "clean inputs", labels: ["Bicubic", "FSRNet clean-trained", "FSRNet degradation-trained", "Real-ESRGAN"], values: am.map((m) => +N.a2bars.bicubic[m].toFixed(1)) },
    { name: "blur + noise + JPEG inputs", labels: ["Bicubic", "FSRNet clean-trained", "FSRNet degradation-trained", "Real-ESRGAN"], values: am.map((m) => +N.a2bars.mixed[m].toFixed(1)) },
  ], Object.assign(chartBase("A2 · PSNR-Y (dB), 3×, all 66 images"), {
    x: 0.6, y: 1.3, w: 6.6, h: 4.5, barDir: "col", chartColors: [HEX.ours === "" ? "" : "9EC3EF", "1F5FA8"],
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.0", valAxisMinVal: 27, valAxisMaxVal: 35, valAxisHidden: true, barGapWidthPct: 50,
  }));
  const ab = [
    ["A1 · L1 vs MSE loss", `${sgn(N.a1.mean)} dB for L1 — consistent but tiny`, C.accent5],
    ["A2 · train on degraded inputs", `${sgn(N.a2m.mean)} dB on degraded, ${sgn(N.a2c.mean)} dB on clean`, C.accent1],
    ["A3 · add 38 poster images", `${sgn(N.a3p.mean)} dB on posters (better on ${Math.round(N.a3p.win * 100)}%), photos unchanged`, C.accent1],
    ["Architecture · vs SRCNN", `${sgn(N.arch.mean)} dB, ${Math.round(N.mem.SRCNN / N.mem["FSRNet-L1"])}× less GPU memory`, C.accent1],
  ];
  ab.forEach(([h, d, col], i) => {
    const y = 1.35 + i * 1.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.5, y, w: 5.2, h: 1.2, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: "ablation " + i });
    txt(s, h, { x: 7.75, y: y + 0.12, w: 4.8, h: 0.4, fontSize: 16, bold: true, color: HEX.dark });
    txt(s, d, { x: 7.75, y: y + 0.58, w: 4.8, h: 0.5, fontSize: 14, color: col });
  });
  txt(s, "One factor changed per ablation; everything else identical to the main model.", { x: 0.6, y: 6.15, w: 6.6, h: 0.4, fontSize: 12, color: HEX.muted });
  s.addNotes(`[7:15–8:00] Ablations change one factor at a time. The loss barely matters: L1 beats MSE by ${f2(N.a1.mean)} dB. The degradation model matters a lot: training on blur, noise and JPEG gains ${f2(N.a2m.mean)} dB on degraded inputs but loses ${f2(Math.abs(N.a2c.mean))} dB on clean ones — no free robustness. Adding just 38 posters gives a small but consistent gain on posters. And our architecture beats SRCNN while using far less memory.`);

  // ================= 11. Invented detail
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Analysis" });
  s.addText("Failure case: a GAN invents plausible but wrong texture", { placeholder: "title" });
  await pic(s, FIG + "slide_poster_1166038_text_logo_mixed.png", 0.6, 1.3, 7.6, 5.55);
  bullets(s, [
    "Input: blur + noise + JPEG (3×)",
    "Our degradation-trained model restores the lettering and keeps the hexagonal grille",
    "Real-ESRGAN replaces it with a rectangular mesh — sharp, believable, and false",
    "Clean-trained CNNs fail the other way: they sharpen JPEG blocks already in the input",
    `On degraded inputs Real-ESRGAN is still perceptually best (LPIPS ${N.lpips_mixed["Real-ESRGAN"].toFixed(3)} vs ${N.lpips_mixed["FSRNet-L1-mixed"].toFixed(3)})`,
  ], { x: 8.5, y: 1.45, w: 4.2, h: 5.3, fontSize: 15 });
  s.addNotes("[8:00–8:40] A failure case for each side. With a degraded input, Real-ESRGAN produces a crisp but different texture — the hexagonal grille becomes rectangular. That is hallucination: plausible, not true. Our clean-trained networks fail differently: they sharpen compression artefacts. Matching training degradations to the real input is what fixes it.");

  // ================= 12. 3x vs 6x + runtime + limitations
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Analysis" });
  s.addText("8K from 720p is a different regime — and speed matters", { placeholder: "title" });
  await pic(s, FIG + "chart_3x_vs_6x.png", 0.6, 1.3, 6.2, 3.2);
  const rows = [
    [{ text: "720p input", options: { bold: true } }, { text: "→ 4K", options: { bold: true } }, { text: "→ 8K", options: { bold: true } }, { text: "GPU MB", options: { bold: true } }],
    ["Bicubic (CPU)", f2(N.runtime["Bicubic|3"]) + " s", f2(N.runtime["Bicubic|6"]) + " s", "–"],
    ["FSRNet (ours)", f2(N.runtime["FSRNet-L1|3"]) + " s", f2(N.runtime["FSRNet-L1|6"]) + " s", Math.round(N.mem["FSRNet-L1"]).toString()],
    ["SRCNN", f2(N.runtime["SRCNN|3"]) + " s", f2(N.runtime["SRCNN|6"]) + " s", Math.round(N.mem.SRCNN).toString()],
    ["Real-ESRGAN", f2(N.runtime["Real-ESRGAN|3"]) + " s", f2(N.runtime["Real-ESRGAN|6"]) + " s", Math.round(N.mem["Real-ESRGAN"]).toString()],
  ];
  s.addTable(rows, { x: 7.1, y: 1.4, w: 5.6, colW: [2.0, 1.15, 1.15, 1.3], fontSize: 13, color: C.text1, fontFace: "Calibri",
    border: { type: "solid", pt: 0.75, color: "D8D7D3" }, fill: { color: C.background1 }, rowH: 0.42 });
  txt(s, "Tesla T4, median of 3 runs, model loading excluded", { x: 7.1, y: 3.65, w: 5.6, h: 0.3, fontSize: 11, color: HEX.muted });
  bullets(s, [
    `Every method loses 3–5 dB from 3× to 6×: ours ${f1(N.x36["FSRNet-L1"][3] - N.x36["FSRNet-L1"][6])} dB — 8K pixels ≠ 8K information`,
    `Applying our 3× model twice beats model + bicubic by ${sgn(N.six.mean)} dB at ~4× the time`,
    `FSRNet is ~${Math.round(N.runtime["Real-ESRGAN|3"] / N.runtime["FSRNet-L1|3"])}× faster than Real-ESRGAN at 720p→4K`,
    "Limits: 16 test posters, synthetic degradations, single training seed, LPIPS is a proxy",
  ], { x: 0.6, y: 4.75, w: 12.1, h: 2.1, fontSize: 15 });
  s.addNotes("[8:40–9:20] At 6x all methods drop by three to five decibels — an 8K image from 720p simply does not contain 8K worth of real information. Staged inference helps a little. Our model runs in about half a second for 720p to 4K, roughly 26 times faster than Real-ESRGAN. Limitations: a small poster test set, synthetic degradations, a single training seed, and LPIPS is only a proxy for human judgement.");

  // ================= 13. Conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("Conclusions: faithful beats flashy for artwork", { placeholder: "title" });
  const speed = Math.round(N.runtime["Real-ESRGAN|3"] / N.runtime["FSRNet-L1|3"]);
  stat(s, sgn(N.e2p.mean, 1) + " dB", "FSRNet over bicubic on posters (better on every test image)", 0.6, 1.4, 3.9, C.accent1);
  stat(s, sgn(N.e3p.mean, 1) + " dB", "FSRNet over Real-ESRGAN on posters — and lower LPIPS", 4.7, 1.4, 3.9, C.accent1);
  stat(s, "~" + speed + "×", "faster than Real-ESRGAN at 720p→4K, with 0.37 M parameters", 8.8, 1.4, 3.9, C.accent1);
  const take = [
    ["1", "Sharpness ≠ accuracy", "The GAN wins perceptually on degraded photos, partly by inventing texture and thickening text."],
    ["2", "Match the degradations", "Training on blur + noise + JPEG is worth more than any loss choice — but costs accuracy on clean inputs."],
    ["3", "Future work", "Fidelity-constrained perceptual fine-tuning, a text-aware loss, a dedicated 6× model and a human study."],
  ];
  take.forEach(([n, h, d], k) => {
    const x = 0.6 + k * 4.1, y = 3.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.9, h: 2.3, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, objectName: "takeaway " + n });
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fill: { color: k === 2 ? C.accent2 : C.accent1 }, line: { color: k === 2 ? C.accent2 : C.accent1 }, objectName: "take num " + n });
    txt(s, n, { x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, h, { x: x + 1.0, y: y + 0.3, w: 2.75, h: 0.5, fontSize: 19, bold: true, color: HEX.dark, valign: "middle" });
    txt(s, d, { x: x + 0.25, y: y + 1.05, w: 3.4, h: 1.55, fontSize: 15, color: C.text1 });
  });
  txt(s, "Thank you — questions?  ·  Live demo: Gradio GUI (scripts/launch_gui.py)", { x: 0.6, y: 6.05, w: 12.1, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
  s.addNotes("[9:20–10:00] To conclude: for faithful enlargement of artwork, a small network trained for the task beats both interpolation and a popular GAN on accuracy, and on posters even perceptually. GANs look great on degraded photos but partly by invention. And the biggest practical lever is matching training degradations to real inputs. Thank you — happy to take questions, and we can show the GUI live.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
