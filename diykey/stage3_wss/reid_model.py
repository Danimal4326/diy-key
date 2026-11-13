# SPDX-License-Identifier: AGPL-3.0-or-later
"""Person re-identification vectors (OSNet-AIN x1.0, MSMT17) on Apple's GPU.

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
"""
import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger("diykey.reid")

REID_DIM = 1024
MODEL_NAME = "diykey-osnet-ain-x1-0-msmt17"
_SIZE = (128, 256)  # width, height the weights were trained on
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_lock = threading.Lock()
_loaded = None  # (model, device, torch) once loaded; False after a failed load


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


_loaded_key = None
_load_lock = threading.Lock()


def _load():
    global _loaded, _loaded_key
    path = os.path.expanduser(os.environ.get("DIYKEY_REID_MODEL", "").strip())
    key = (path, os.path.getmtime(path) if os.path.exists(path) else None)
    with _load_lock:
        if _loaded is False and _loaded_key != key:
            _loaded = None
        if _loaded is not None:
            return _loaded
        _loaded_key = key
        try:
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state = {k.removeprefix("module."): v
                     for k, v in ckpt.get("state_dict", ckpt).items()
                     if not k.removeprefix("module.").startswith("classifier")}
            model = _build()
            model.load_state_dict(state, strict=True)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            _loaded = (model.eval().to(device), device, torch)
            log.info("re-ID: %s on %s", os.path.basename(path)[:32], device)
        except Exception as e:  # noqa: BLE001 - detection carries on without re-ID vectors
            log.warning("re-ID model %s failed to load: %r", path, e)
            _loaded = False
        return _loaded


def embed_raw(img_bgr):
    """The model's own vector (OSNet is 512) for a person crop (BGR), or None."""
    if not enabled() or img_bgr is None or img_bgr.size == 0:
        return None
    loaded = _load()
    if not loaded:
        return None
    model, device, torch = loaded
    try:
        rgb = cv2.cvtColor(cv2.resize(img_bgr, _SIZE, interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with _lock, torch.no_grad():
            v = model(tensor).float().cpu().numpy().reshape(-1)
    except Exception as e:  # noqa: BLE001 - one bad crop must not lose the event's other tags
        log.warning("re-ID failed on a %s crop: %r", img_bgr.shape[:2], e)
        return None
    if not np.isfinite(v).all():
        return None
    return [float(f) for f in v]


def embed(img_bgr):
    """Zero-padded to REID_DIM for ramDetections.reidEmbedding, or None."""
    raw = embed_raw(img_bgr)
    if raw is None:
        return None
    return raw[:REID_DIM] + [0.0] * max(0, REID_DIM - len(raw))
