# SPDX-License-Identifier: AGPL-3.0-or-later
"""Person re-identification vectors (OSNet-AIN x1.0, MSMT17).

The AI Key computes reidEmbed with SmartReID_encoder.engine, a TensorRT build Protect's own
settings text calls OSNet ("Per-camera OSNet reid cutoff", M/507273:106); it only runs on
NVIDIA hardware. This is the published OSNet-AIN network (Zhou et al., MIT), same family,
trained on 256x128 person crops. Weights: DIYKEY_REID_MODEL, the checkpoint from Hugging Face
kaiyangzhou/osnet, read weights-only.

Protect compares these two ways, so the vectors must be raw, not unit length:
  * tap-to-find (M/909595 reidentifySearch) keeps '"reidEmbedding" <-> :embedding' L2 distance
    <= 20 + sensitivity/10 (M/772678:70, similarity = 100 - distance). Firmware output is
    unnormalised (trt_engine.py encode() has the normalise commented out; PackEmbed copies it).
    Unit vectors are at most 2 apart, so every person would match.
  * dedup sessions (M/374992 matchSession) use cosine against reidCutoffDefault 0.75.
Protect's column is 1024-d (M/315148:83); OSNet gives 512, so the rest is zero-padded, which
leaves both L2 distance and cosine unchanged.

Two backends, by DIYKEY_REID_MODEL's file type:
  * .pth   the checkpoint itself, in torch (Apple GPU through MPS, else CPU)
  * .onnx  scripts/export_reid_onnx.py's export of it, in onnxruntime: no torch, and on the
           Intel GPU with DIYKEY_ORT_DEVICE (ort_device.py).
Live 2026-11-11 on an i5-1240P (Iris Xe, host under load), 32 real person crops: torch CPU
~200 ms, onnxruntime CPU ~92 ms, OpenVINO GPU ~20 ms a crop (first load ~8 s to compile, then
cached). The ONNX vectors equal torch's on CPU (L2 < 0.001); the GPU runs FP16: cosine >= 0.9997,
L2 <= 0.8, against vector norms of 30-46, pairwise distances of 9-29 (median) and Protect's
tap-to-find cut of 20-30.
"""
import logging
import os
import threading

import cv2
import numpy as np

import ort_device

log = logging.getLogger("diykey.reid")

REID_DIM = 1024
MODEL_NAME = "diykey-osnet-ain-x1-0-msmt17"
_SIZE = (128, 256)  # width, height the weights were trained on
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_lock = threading.Lock()
_loaded = None  # run(x) -> (1, 512) array once loaded; False after a failed load


def enabled():
    return bool(os.environ.get("DIYKEY_REID_MODEL", "").strip())


def _build():
    from torch import nn
    import torch.nn.functional as F

    class ConvLayer(nn.Module):
        def __init__(self, cin, cout, k, stride=1, padding=0, instance=False):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
            self.bn = nn.InstanceNorm2d(cout, affine=True) if instance else nn.BatchNorm2d(cout)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Conv1x1(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn = nn.BatchNorm2d(cout)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Conv1x1Linear(nn.Module):
        def __init__(self, cin, cout, bn=True):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn = nn.BatchNorm2d(cout) if bn else None

        def forward(self, x):
            x = self.conv(x)
            return self.bn(x) if self.bn is not None else x

    class LightConv3x3(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
            self.bn = nn.BatchNorm2d(cout)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv2(self.conv1(x))))

    class LightConvStream(nn.Module):
        def __init__(self, cin, cout, depth):
            super().__init__()
            self.layers = nn.Sequential(LightConv3x3(cin, cout),
                                        *[LightConv3x3(cout, cout) for _ in range(depth - 1)])

        def forward(self, x):
            return self.layers(x)

    class ChannelGate(nn.Module):
        def __init__(self, c, reduction=16):
            super().__init__()
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(c, c // reduction, 1)
            self.relu = nn.ReLU()
            self.fc2 = nn.Conv2d(c // reduction, c, 1)
            self.gate_activation = nn.Sigmoid()

        def forward(self, x):
            w = self.fc2(self.relu(self.fc1(self.global_avgpool(x))))
            return x * self.gate_activation(w)

    class OSBlock(nn.Module):
        instance = False

        def __init__(self, cin, cout, reduction=4, T=4):
            super().__init__()
            mid = cout // reduction
            self.conv1 = Conv1x1(cin, mid)
            self.conv2 = nn.ModuleList(LightConvStream(mid, mid, t) for t in range(1, T + 1))
            self.gate = ChannelGate(mid)
            self.conv3 = Conv1x1Linear(mid, cout, bn=not self.instance)
            self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
            if self.instance:
                self.IN = nn.InstanceNorm2d(cout, affine=True)

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = sum(self.gate(stream(x1)) for stream in self.conv2)
            x3 = self.conv3(x2)
            if self.instance:
                x3 = self.IN(x3)
            identity = self.downsample(x) if self.downsample is not None else x
            return F.relu(x3 + identity)

    class OSBlockINin(OSBlock):
        instance = True

    def stage(kinds, cin, cout):
        return nn.Sequential(kinds[0](cin, cout), *[k(cout, cout) for k in kinds[1:]])

    class OSNetAIN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3, instance=True)
            self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
            self.conv2 = stage([OSBlockINin, OSBlockINin], 64, 256)
            self.pool2 = nn.Sequential(Conv1x1(256, 256), nn.AvgPool2d(2, stride=2))
            self.conv3 = stage([OSBlock, OSBlockINin], 256, 384)
            self.pool3 = nn.Sequential(Conv1x1(384, 384), nn.AvgPool2d(2, stride=2))
            self.conv4 = stage([OSBlockINin, OSBlock], 384, 512)
            self.conv5 = Conv1x1(512, 512)
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU())

        def forward(self, x):
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.conv5(self.conv4(x))
            return self.fc(self.global_avgpool(x).flatten(1))

    return OSNetAIN()


def state_dict(torch, path):
    """The checkpoint's weights for _build(), without the training-only classifier."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return {k.removeprefix("module."): v for k, v in ckpt.get("state_dict", ckpt).items()
            if not k.removeprefix("module.").startswith("classifier")}


def preprocess(img_bgr):
    """(1, 3, 256, 128) float32: resized, RGB, ImageNet-normalised, as the weights expect."""
    rgb = cv2.cvtColor(cv2.resize(img_bgr, _SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def _torch_runner(path):
    import torch
    model = _build()
    model.load_state_dict(state_dict(torch, path), strict=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.eval().to(device)

    def run(x):
        with torch.no_grad():
            return model(torch.from_numpy(x).to(device)).float().cpu().numpy()
    return run, f"torch {device}"


def _onnx_runner(path):
    import onnxruntime as ort
    sess = ort_device.session(ort, path, ["CPUExecutionProvider"], "re-ID")
    name = sess.get_inputs()[0].name

    def run(x):
        return sess.run(None, {name: x})[0]
    return run, ort_device.describe(sess)


def _load():
    global _loaded
    if _loaded is not None:
        return _loaded
    path = os.path.expanduser(os.environ.get("DIYKEY_REID_MODEL", "").strip())
    try:
        runner = _onnx_runner if path.lower().endswith(".onnx") else _torch_runner
        run, where = runner(path)
        _loaded = run
        log.info("re-ID: %s on %s", os.path.basename(path)[:32], where)
    except Exception as e:  # noqa: BLE001 - detection carries on without re-ID vectors
        log.warning("re-ID model %s failed to load: %r", path, e)
        _loaded = False
    return _loaded


def embed(img_bgr):
    """Raw vector of REID_DIM floats for a person crop (BGR), or None."""
    if not enabled() or img_bgr is None or img_bgr.size == 0:
        return None
    loaded = _load()
    if not loaded:
        return None
    x = preprocess(img_bgr)
    with _lock:
        v = np.asarray(loaded(x), np.float32).reshape(-1)
    if not np.isfinite(v).all():
        return None
    return [float(f) for f in v] + [0.0] * (REID_DIM - v.size)
