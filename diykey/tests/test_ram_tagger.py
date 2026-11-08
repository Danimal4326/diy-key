# SPDX-License-Identifier: AGPL-3.0-or-later
"""build_result_with_images: tagObjs and searchSnapshots from roiMeta (SW:2737-2760, 4062-4070)."""
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ram_tagger  # noqa: E402


class FakeTagger:
    def tag_frame(self, img):
        return [{"tag": "person", "confScore": 0.9}]


class RamImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "f.jpg")
        cv2.imwrite(self.path, np.full((200, 400, 3), 120, np.uint8))

    def test_tagobjs_and_snapshots(self):
        roi = [{"coord": [100, 100, 300, 600], "objectType": "person", "trackerId": 5},
               {"coord": [0, 0, 0, 0], "objectType": "vehicle", "trackerId": 9},
               {"coord": [1, 2, 3, 4], "objectType": "weird", "trackerId": 11}]
        meta = [{"ts": 1000, "roi": roi}, {"ts": 2000, "roi": roi[0]}]
        res, imgs = ram_tagger.build_result_with_images(
            FakeTagger(), "cam", "ev", {1000: self.path}, roi_meta=meta)
        km, = res["keyMomentsTags"]
        self.assertEqual(km["tagObjs"], roi)
        snaps = {s["trackerID"]: s for s in km["searchSnapshots"]}
        self.assertEqual(set(snaps), {5, 9})  # unknown type skipped
        self.assertEqual(set(imgs), {"5", "9"})
        s = snaps[5]
        self.assertEqual((s["clockBestWall"], s["smartDetectSnapshotType"]), (1000, "person"))
        self.assertEqual(s["smartDetectSnapshot"], "/tmp/cam_5.jpg")
        crop = cv2.imdecode(np.frombuffer(imgs["5"], np.uint8), 1)
        self.assertLess(crop.shape[1], 400)           # roi crop
        full = cv2.imdecode(np.frombuffer(imgs["9"], np.uint8), 1)
        self.assertEqual(full.shape[:2], (200, 400))  # unusable box -> whole frame

    def test_build_result_unchanged(self):
        res = ram_tagger.build_result(FakeTagger(), "cam", "ev", {1000: self.path})
        self.assertIsInstance(res, dict)
        self.assertNotIn("tagObjs", res["keyMomentsTags"][0])
        self.assertNotIn("searchSnapshots", res["keyMomentsTags"][0])


if __name__ == "__main__":
    unittest.main()
