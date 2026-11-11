#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export the person re-ID model to ONNX, for DIYKEY_REID_MODEL=<file>.onnx (no torch at runtime).

Reads the OSNet-AIN x1.0 MSMT17 checkpoint (Hugging Face kaiyangzhou/osnet, MIT; the .pth
reid_model's torch backend loads) and writes reid_model's network with those weights as ONNX
(~9 MB): input "pixels" (N, 3, 256, 128) from reid_model.preprocess, output "embedding" (N, 512),
raw like the torch model (Protect compares L2 distances, so nothing is normalised). Then it
checks the export against torch on random and, if given, real crops.

Needs, on the machine that runs it (not the AI Key): torch, onnx, onnxruntime,
opencv-python-headless, numpy.

  python3 scripts/export_reid_onnx.py osnet_ain_x1_0_msmt17_....pth /path/to/models/osnet-ain-x1-0-msmt17.onnx [crop.png ...]
  # then DIYKEY_REID_MODEL=/path/to/models/osnet-ain-x1-0-msmt17.onnx (and DIYKEY_ORT_DEVICE=GPU)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "diykey"), os.path.join(ROOT, "diykey", "stage3_wss")]


def main(checkpoint, out, crops):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    import reid_model

    model = reid_model._build()
    model.load_state_dict(reid_model.state_dict(torch, checkpoint), strict=True)
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (h, w, 3), np.uint8) for h, w in ((300, 120), (64, 40))]
    imgs += [cv2.imread(c) for c in crops]
    torch.onnx.export(model, (torch.from_numpy(reid_model.preprocess(imgs[0])),), out,
                      input_names=["pixels"], output_names=["embedding"],
                      dynamic_axes={"pixels": {0: "batch"}, "embedding": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    worst_cos, worst_l2 = 1.0, 0.0
    with torch.no_grad():
        for img in imgs:
            x = reid_model.preprocess(img)
            want = model(torch.from_numpy(x)).numpy()[0]
            got = sess.run(None, {"pixels": x})[0][0]
            worst_cos = min(worst_cos, float(want @ got / np.linalg.norm(want) / np.linalg.norm(got)))
            worst_l2 = max(worst_l2, float(np.linalg.norm(want - got)))
    assert worst_cos > 0.9999 and worst_l2 < 0.5, (worst_cos, worst_l2)
    print(f"checked against torch on {len(imgs)} crops: worst cosine {worst_cos:.6f}, "
          f"L2 {worst_l2:.4f}. Set DIYKEY_REID_MODEL={os.path.abspath(out)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
