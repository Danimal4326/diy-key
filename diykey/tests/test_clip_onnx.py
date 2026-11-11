# SPDX-License-Identifier: AGPL-3.0-or-later
"""clip_embed's ONNX backend (DIYKEY_CLIP_ONNX): open_clip-exact preprocessing, open_clip token
framing, backend choice and failure, without the 1.7 GB model files."""
import importlib.util
import os
import sys
import unittest
from unittest import mock

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import clip_embed  # noqa: E402

_PIL = importlib.util.find_spec("PIL") is not None
_OPEN_CLIP = _PIL and all(importlib.util.find_spec(m) for m in ("open_clip", "torch"))


class FakeTokenizer:
    """tokenizers.Tokenizer stand-in: one id per character."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return self._Enc([100 + ord(c) % 50 for c in text])


class TokensTest(unittest.TestCase):
    def test_open_clip_framing(self):
        row = clip_embed.tokens(FakeTokenizer(), "car")
        self.assertEqual(row.shape, (1, 77))
        self.assertEqual(row.dtype, np.int64)
        self.assertEqual(row[0, :5].tolist(), [clip_embed.SOT, 149, 147, 114, clip_embed.EOT])
        self.assertEqual(set(row[0, 5:].tolist()), {0})

    def test_long_text_truncated_keeping_the_end_token(self):
        row = clip_embed.tokens(FakeTokenizer(), "x" * 200)
        self.assertEqual(row[0, 0], clip_embed.SOT)
        self.assertEqual(row[0, 76], clip_embed.EOT)
        self.assertNotIn(0, row[0].tolist())


@unittest.skipUnless(_PIL, "Pillow not installed")
class PreprocessTest(unittest.TestCase):
    def _images(self):
        rng = np.random.default_rng(3)
        return [rng.integers(0, 256, (h, w, 3), np.uint8)
                for h, w in ((1512, 2688), (100, 37), (37, 100), (224, 224), (225, 300), (63, 63))]

    def test_shape_and_type(self):
        for img in self._images():
            x = clip_embed.preprocess(img)
            self.assertEqual((x.shape, x.dtype), ((1, 3, 224, 224), np.float32))

    @unittest.skipUnless(_OPEN_CLIP, "open_clip/torch not installed")
    def test_identical_to_open_clip(self):
        # Live 2026-11-10: max difference 0 over 29 real frames, crops and odd sizes.
        import cv2
        import open_clip
        from PIL import Image
        _, _, ref = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained=None)
        for img in self._images():
            want = ref(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).numpy()
            np.testing.assert_array_equal(clip_embed.preprocess(img)[0], want)


class FakeOnnxBackend:
    made = []

    def __init__(self, folder):
        FakeOnnxBackend.made.append(folder)
        self.where = "ONNX, image on OpenVINO GPU, text on OpenVINO GPU"
        self.texts = []

    def image(self, img_bgr):
        return np.array([[3.0, 4.0] + [0.0] * 766], np.float32)

    def text(self, phrase):
        self.texts.append(phrase)
        return np.array([[0.0, 0.0, 2.0] + [0.0] * 765], np.float32)


class BackendChoiceTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"DIYKEY_CLIP": "1",
                                               "DIYKEY_CLIP_ONNX": "/models/clip"})
        patcher.start()
        self.addCleanup(patcher.stop)
        clip_embed._model = None
        self.addCleanup(setattr, clip_embed, "_model", None)
        FakeOnnxBackend.made.clear()

    def test_onnx_dir_selects_the_onnx_backend(self):
        with mock.patch.object(clip_embed, "_OnnxBackend", FakeOnnxBackend), \
                mock.patch.object(clip_embed, "_TorchBackend", side_effect=AssertionError("torch")), \
                self.assertLogs("diykey.clip", "INFO") as logs:
            img = clip_embed.image_embed(np.zeros((8, 8, 3), np.uint8))
            txt = clip_embed.text_embed("car")
        self.assertEqual(FakeOnnxBackend.made, ["/models/clip"])
        self.assertIn("OpenVINO GPU", logs.output[0])
        np.testing.assert_allclose(img[:2], [0.6, 0.8], rtol=1e-6)   # unit length
        self.assertEqual(txt[2], 1.0)
        self.assertEqual(clip_embed._model.texts, ["a photo of car"])  # short query prefix

    def test_without_the_setting_the_torch_backend_loads(self):
        del os.environ["DIYKEY_CLIP_ONNX"]
        torch_backend = FakeOnnxBackend("torch")
        with mock.patch.object(clip_embed, "_TorchBackend", return_value=torch_backend) as tb, \
                mock.patch.object(clip_embed, "_OnnxBackend", side_effect=AssertionError("onnx")):
            clip_embed.text_embed("a dog in the yard")
        tb.assert_called_once_with()

    def test_broken_onnx_dir_disables_clip_with_a_warning(self):
        with mock.patch.object(clip_embed, "_OnnxBackend", side_effect=FileNotFoundError("visual")), \
                self.assertLogs("diykey.clip", "WARNING") as logs:
            self.assertIsNone(clip_embed.image_embed(np.zeros((8, 8, 3), np.uint8)))
            self.assertIsNone(clip_embed.text_embed("car"))   # not retried
        self.assertIn("DIYKEY_CLIP_ONNX=/models/clip", logs.output[0])
        self.assertEqual(len(logs.records), 1)

    def test_off_without_diykey_clip(self):
        os.environ["DIYKEY_CLIP"] = "0"
        with mock.patch.object(clip_embed, "_OnnxBackend", side_effect=AssertionError("loaded")):
            self.assertIsNone(clip_embed.text_embed("car"))


if __name__ == "__main__":
    unittest.main()
