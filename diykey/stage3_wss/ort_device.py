# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the onnxruntime models run: DIYKEY_ORT_DEVICE.

  unset, "" or "cpu"   as before: each stage's own providers (CPU, plus Core ML where a stage
                       asks for it on macOS)
  an OpenVINO device   onnxruntime-openvino's OpenVINOExecutionProvider on that device (GPU,
                       GPU.1, NPU, CPU, AUTO:GPU,CPU, ...), CPU after it

OpenVINO needs the onnxruntime-openvino wheel in place of onnxruntime (docker/Dockerfile
ORT=openvino), and for an Intel GPU the OpenCL compute runtime plus /dev/dri. Without the
provider the setting is ignored with one warning; a device that can't compile a model leaves
that stage on its own providers, logged. Used by the COCO detector, RAM++, face enhancement and
CLIP's ONNX backend (clip_embed.py, DIYKEY_CLIP_ONNX).

Live 2026-11-10 on an i5-1240P (Iris Xe), the COCO detector per real key-moment frame:
YOLOv9-s 320 px 22.6 -> 13.2 ms, YOLO11m 640 px 202.9 -> 63.5 ms, YOLOv9-c 640 px 350.2 ->
83.4 ms, tags identical on 33-38 of 40 frames (score changes <= 0.033, the GPU runs FP16; the
rest were tags at the 0.35 cut-off). First load ~3 s to compile; the compiled model is cached
in DIYKEY_ORT_CACHE_DIR (default <XDG_CACHE_HOME or ~/.cache>/diykey-openvino).

lpr.py (fast-alpr) is left on CPU: fast-alpr makes its own sessions, so this fallback can't
cover them, and its models already fail under Core ML (dynamic outputs).
"""
import logging
import os

log = logging.getLogger("diykey.ort")

ENV = "DIYKEY_ORT_DEVICE"
OPENVINO = "OpenVINOExecutionProvider"
CPU = "CPUExecutionProvider"
_warned = set()


def openvino_device():
    """The OpenVINO device DIYKEY_ORT_DEVICE names, or None for the stages' own providers."""
    value = (os.environ.get(ENV) or "").strip()
    return None if value.lower() in ("", "cpu", "off", "0") else value.upper()


def cache_dir():
    if os.environ.get("DIYKEY_ORT_CACHE_DIR"):
        return os.environ["DIYKEY_ORT_CACHE_DIR"]
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "diykey-openvino")


def providers(ort, base):
    """Providers for a session: `base` (the stage's own list), or OpenVINO on the configured
    device first and then CPU."""
    device = openvino_device()
    if device is None:
        return list(base)
    if OPENVINO not in ort.get_available_providers():
        if "no_provider" not in _warned:
            _warned.add("no_provider")
            log.warning("%s=%s but this onnxruntime has no %s (install onnxruntime-openvino); "
                        "models stay on %s", ENV, device, OPENVINO, ", ".join(_names(base)))
        return list(base)
    options = {"device_type": device}
    cache = cache_dir()
    try:
        os.makedirs(cache, exist_ok=True)
        options["cache_dir"] = cache
    except OSError as exc:
        log.warning("OpenVINO model cache %s unusable (%s); compiling uncached", cache, exc)
    return [(OPENVINO, options), CPU]


def session(ort, path, base, label):
    """InferenceSession for `path` on providers(); if OpenVINO can't compile it on the device,
    one on `base` instead."""
    chosen = providers(ort, base)
    try:
        sess = ort.InferenceSession(path, providers=chosen)
    except Exception:
        if _names(chosen) == _names(base):
            raise
        log.exception("%s: OpenVINO on %s failed for %s; using %s", label, openvino_device(),
                      path, ", ".join(_names(base)))
        return ort.InferenceSession(path, providers=list(base))
    # onnxruntime itself drops a provider it can't create (e.g. a device that isn't there:
    # live 2026-11-10, GPU.7) with only a stderr line, and runs the session on the rest.
    if OPENVINO in _names(chosen) and sess.get_providers()[0] != OPENVINO:
        log.warning("%s: OpenVINO on %s isn't available; %s runs on %s", label,
                    openvino_device(), path, sess.get_providers()[0])
    return sess


def describe(sess):
    """'OpenVINO GPU' / 'CPUExecutionProvider' / ... for a log line."""
    first = sess.get_providers()[0]
    if first == OPENVINO:
        opts = (sess.get_provider_options() or {}).get(OPENVINO, {})
        return f"OpenVINO {opts.get('device_type') or openvino_device() or ''}".strip()
    return first


def _names(provs):
    return [p[0] if isinstance(p, tuple) else p for p in provs]
