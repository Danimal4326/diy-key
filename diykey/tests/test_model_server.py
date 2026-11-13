# SPDX-License-Identifier: AGPL-3.0-or-later
"""model_server: the :7788 SSV placeholder accepts one roi object per thumbnail."""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from stage4_modelserver import model_server  # noqa: E402


class RoiShapeTest(unittest.TestCase):
    def test_single_roi_object(self):
        out = model_server._second_verifier_result({
            "camera": "c", "event": "e",
            "thumbnailMeta": [{"ts": 1, "roi": {"objectType": "person", "trackerID": 7}}]})
        self.assertEqual(out["verificationResults"][0]["trackerID"], 7)
        self.assertEqual(out["verificationResults"][0]["objectType"], "person")


if __name__ == "__main__":
    unittest.main()
