# SPDX-License-Identifier: AGPL-3.0-or-later
"""clip_embed: DIYKEY_CLIP_FP16 is opt-in and GPU-only (Mac B 2026-11-10: FP16 moved Protect's
similarity by <= 0.28 points on 44 Driveway crops x 15 queries, +889 MB vs +2.5 GB FP32), and
Protect's similarity mapping."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

from stage3_wss import clip_embed as ce  # noqa: E402


class Fp16ToggleTest(unittest.TestCase):
    def _env(self, **kw):
        env = {k: v for k, v in os.environ.items()
               if k not in ("DIYKEY_CLIP_FP16", "DIYKEY_CLIP_FP16_WEIGHTS")}
        env.update(kw)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_off_by_default(self):
        with self._env():
            self.assertFalse(ce.fp16_wanted("mps"))

    def test_on_for_gpu_devices_only(self):
        with self._env(DIYKEY_CLIP_FP16="1"):
            self.assertTrue(ce.fp16_wanted("mps"))
            self.assertTrue(ce.fp16_wanted("cuda"))
            self.assertFalse(ce.fp16_wanted("cpu"))

    def test_only_1_turns_it_on(self):
        with self._env(DIYKEY_CLIP_FP16="true"):
            self.assertFalse(ce.fp16_wanted("mps"))

    def test_weights_path_default_and_override(self):
        with self._env():
            self.assertEqual(ce.fp16_weights_path(),
                             os.path.expanduser("~/.diy-key/models/" + ce.FP16_FILE))
        with self._env(DIYKEY_CLIP_FP16_WEIGHTS="~/w/clip16.safetensors"):
            self.assertEqual(ce.fp16_weights_path(), os.path.expanduser("~/w/clip16.safetensors"))


class EnabledTest(unittest.TestCase):
    def _env(self, value):
        return mock.patch.dict(os.environ, {"DIYKEY_CLIP": value}, clear=False)

    def test_enabled_accepts_the_same_truthy_forms_as_the_feature_flag(self):
        for value in ("1", "true", "yes", "on", "TRUE"):
            with self._env(value):
                self.assertTrue(ce.enabled(), value)
        for value in ("0", "false", "no", "off", ""):
            with self._env(value):
                self.assertFalse(ce.enabled(), value)


class SimilarityTest(unittest.TestCase):
    def test_protect_mapping(self):
        a = np.array([1.0, 0.0])
        self.assertAlmostEqual(ce.protect_similarity(a, a), 100 * 0.92 / 0.32)
        b = np.array([np.cos(np.arccos(0.24)), np.sin(np.arccos(0.24))])
        self.assertAlmostEqual(ce.protect_similarity(a, b), 50.0)


if __name__ == "__main__":
    unittest.main()
