# Models

* `*.pt` — our trained checkpoints (copied from `outputs/runs/<run>/best.pt`). Not in git; attach them to a
  GitHub release or share the Drive folder. Expected names (see `configs/eval.yaml`):
  `fsrnet_x3_l1_bicubic.pt`, `fsrnet_x3_mse_bicubic.pt`, `fsrnet_x3_l1_mixed.pt`, `srcnn_x3_l1_bicubic.pt`.
* `pretrained/` — third-party files fetched explicitly by `python scripts/download_weights.py`:
  `RealESRGAN_x4plus.pth` (BSD-3-Clause, https://github.com/xinntao/Real-ESRGAN) and the OpenCV
  frontal-face Haar cascade (Apache-2.0) used for automatic face ROIs.
