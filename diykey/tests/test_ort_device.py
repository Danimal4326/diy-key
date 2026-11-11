# SPDX-License-Identifier: AGPL-3.0-or-later
"""ort_device: DIYKEY_ORT_DEVICE picks OpenVINO's provider, else each stage's own; fallback."""
import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ort_device  # noqa: E402

CPU = ["CPUExecutionProvider"]
COREML = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}), "CPUExecutionProvider"]


class FakeOrt:
    """onnxruntime stand-in: available providers, and sessions that may fail on OpenVINO."""

    def __init__(self, available, openvino_fails=False):
        self.available = available
        self.openvino_fails = openvino_fails
        self.calls = []

    def get_available_providers(self):
        return list(self.available)

    def InferenceSession(self, path, providers=None):  # noqa: N802 - onnxruntime's name
        self.calls.append(providers)
        names = [p[0] if isinstance(p, tuple) else p for p in providers]
        if "OpenVINOExecutionProvider" in names and self.openvino_fails:
            raise RuntimeError("GPU device not found")
        return mock.Mock(providers=providers)


class ProvidersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = {k: v for k, v in os.environ.items() if not k.startswith("DIYKEY_ORT")}
        env["DIYKEY_ORT_CACHE_DIR"] = os.path.join(self.tmp.name, "cache")
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        ort_device._warned.clear()

    def test_unset_or_cpu_keeps_each_stages_own(self):
        ort = FakeOrt(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        for value in (None, "", "cpu", "CPU", "off"):
            if value is None:
                os.environ.pop("DIYKEY_ORT_DEVICE", None)
            else:
                os.environ["DIYKEY_ORT_DEVICE"] = value
            self.assertIsNone(ort_device.openvino_device())
            self.assertEqual(ort_device.providers(ort, CPU), CPU)
            self.assertEqual(ort_device.providers(ort, COREML), COREML)  # macOS stays on Core ML

    def test_openvino_device_first_then_cpu(self):
        os.environ["DIYKEY_ORT_DEVICE"] = "gpu"
        ort = FakeOrt(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        provs = ort_device.providers(ort, COREML)
        self.assertEqual(provs[0][0], "OpenVINOExecutionProvider")
        self.assertEqual(provs[0][1]["device_type"], "GPU")
        self.assertEqual(provs[0][1]["cache_dir"], os.environ["DIYKEY_ORT_CACHE_DIR"])
        self.assertTrue(os.path.isdir(os.environ["DIYKEY_ORT_CACHE_DIR"]))
        self.assertEqual(provs[1:], ["CPUExecutionProvider"])
        os.environ["DIYKEY_ORT_DEVICE"] = "AUTO:GPU,CPU"
        self.assertEqual(ort_device.providers(ort, CPU)[0][1]["device_type"], "AUTO:GPU,CPU")

    def test_cache_defaults_under_xdg_cache_home(self):
        del os.environ["DIYKEY_ORT_CACHE_DIR"]
        os.environ["XDG_CACHE_HOME"] = "/data/cache"
        self.assertEqual(ort_device.cache_dir(), "/data/cache/diykey-openvino")

    def test_without_the_openvino_wheel_warns_once_and_keeps_own(self):
        os.environ["DIYKEY_ORT_DEVICE"] = "GPU"
        ort = FakeOrt(["CPUExecutionProvider"])
        with self.assertLogs("diykey.ort", "WARNING") as logs:
            self.assertEqual(ort_device.providers(ort, CPU), CPU)
            self.assertEqual(ort_device.providers(ort, CPU), CPU)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("onnxruntime-openvino", logs.output[0])

    def test_session_falls_back_when_the_device_cannot_compile(self):
        os.environ["DIYKEY_ORT_DEVICE"] = "GPU"
        ort = FakeOrt(["OpenVINOExecutionProvider", "CPUExecutionProvider"], openvino_fails=True)
        with self.assertLogs("diykey.ort", "ERROR"):
            sess = ort_device.session(ort, "/m.onnx", CPU, "RAM tagger")
        self.assertEqual(sess.providers, CPU)
        self.assertEqual(len(ort.calls), 2)

    def test_session_says_when_onnxruntime_dropped_openvino(self):
        # onnxruntime doesn't raise for a device it can't open: it runs on the next provider.
        os.environ["DIYKEY_ORT_DEVICE"] = "GPU.7"
        ort = FakeOrt(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        dropped = mock.Mock()
        dropped.get_providers.return_value = ["CPUExecutionProvider"]
        ort.InferenceSession = mock.Mock(return_value=dropped)
        with self.assertLogs("diykey.ort", "WARNING") as logs:
            self.assertIs(ort_device.session(ort, "/m.onnx", CPU, "RAM tagger"), dropped)
        self.assertIn("OpenVINO on GPU.7 isn't available", logs.output[0])
        self.assertIn("CPUExecutionProvider", logs.output[0])

    def test_session_on_openvino_says_nothing(self):
        os.environ["DIYKEY_ORT_DEVICE"] = "GPU"
        ort = FakeOrt(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        on_gpu = mock.Mock()
        on_gpu.get_providers.return_value = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        ort.InferenceSession = mock.Mock(return_value=on_gpu)
        with self.assertNoLogs("diykey.ort", "WARNING"):
            ort_device.session(ort, "/m.onnx", CPU, "RAM tagger")

    def test_session_on_own_providers_still_raises(self):
        ort = FakeOrt(["CPUExecutionProvider"])
        ort.InferenceSession = mock.Mock(side_effect=RuntimeError("bad model"))
        with self.assertRaises(RuntimeError):
            ort_device.session(ort, "/m.onnx", CPU, "RAM tagger")

    def test_describe(self):
        sess = mock.Mock()
        sess.get_providers.return_value = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        sess.get_provider_options.return_value = {"OpenVINOExecutionProvider": {"device_type": "GPU"}}
        self.assertEqual(ort_device.describe(sess), "OpenVINO GPU")
        sess.get_providers.return_value = ["CPUExecutionProvider"]
        self.assertEqual(ort_device.describe(sess), "CPUExecutionProvider")


if __name__ == "__main__":
    unittest.main()
