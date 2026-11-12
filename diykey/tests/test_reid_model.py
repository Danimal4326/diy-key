# SPDX-License-Identifier: AGPL-3.0-or-later
"""reid_model: raw, zero-padded OSNet vectors; reidEmbed on people only; segment-video."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ram_tagger  # noqa: E402
import reid_model  # noqa: E402
import ucp4_client  # noqa: E402

ENV = {"DIYKEY_REID_MODEL": "/m.pth"}


class EmbedTest(unittest.TestCase):
    def test_disabled_without_model(self):
        with mock.patch.dict(os.environ, {"DIYKEY_REID_MODEL": ""}):
            self.assertIsNone(reid_model.embed(np.zeros((8, 8, 3), np.uint8)))

    def test_padded_to_protect_width_and_not_normalised(self):
        import torch
        model = reid_model._build().eval()
        for m in model.modules():  # a norm large enough to tell raw from unit length
            if isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.bias, 5.0)
        img = np.random.default_rng(1).integers(0, 255, (300, 120, 3), np.uint8)
        with mock.patch.dict(os.environ, ENV), \
                mock.patch.object(reid_model, "_load", return_value=(model, "cpu", torch)):
            vec = reid_model.embed(img)
        self.assertEqual(len(vec), 1024)
        self.assertEqual(set(vec[512:]), {0.0})
        self.assertGreater(float(np.linalg.norm(vec)), 2.0)  # Protect's limit is L2 20-30


class FakeTagger:
    def __init__(self, dets=()):
        self.dets = list(dets)

    def tag_frame(self, img, plus=True):
        return []

    def detect(self, img):
        return self.dets


class ThumbnailTest(unittest.TestCase):
    def _tags(self, obj_type):
        img = np.zeros((40, 20, 3), np.uint8)
        objects = [{"trackerID": 7, "keyMomentMs": 1000, "img": img,
                    "hint": {"objectType": obj_type, "confidence": 90}}]
        with mock.patch.object(reid_model, "embed", return_value=[1.5] * 1024) as embed, \
                mock.patch.object(ram_tagger, "tag_crop", return_value=[]), \
                mock.patch.object(ram_tagger.clip_embed, "image_embed", return_value=None), \
                mock.patch.object(ram_tagger.clip_embed, "blend", return_value=None):
            return ram_tagger.thumbnail_tags(FakeTagger(), objects)[0], embed

    def test_person_gets_reid(self):
        entry, embed = self._tags("person")
        self.assertEqual(len(entry["reidEmbed"]), 1024)
        embed.assert_called_once()

    def test_vehicle_does_not(self):
        entry, embed = self._tags("vehicle")
        self.assertNotIn("reidEmbed", entry)
        embed.assert_not_called()


class SegmentTest(unittest.TestCase):
    def test_people_only_in_0_1000_coords(self):
        img = np.zeros((500, 1000, 3), np.uint8)
        tagger = FakeTagger([("person", 0.9, (100, 50, 200, 250)), ("car", 0.9, (0, 0, 9, 9)),
                             ("person", 0.4, (5, 5, 50, 50))])
        with mock.patch.object(reid_model, "embed", return_value=[1.0] * 1024):
            res = ucp4_client._segment_results(tagger, img)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["bbox"], [100, 100, 300, 600])
        self.assertEqual(res[0]["objectType"], "person")
        self.assertEqual(res[0]["contours"], [res[0]["contour"]])

    def test_narrows_export_and_posts_results(self):
        post = mock.Mock(return_value=mock.Mock(status_code=200))
        body = {"resUrl": "/up", "payload": {"cameraId": "c", "timestamp": 10000,
                "videoUrl": "/internal/aiprocessors/video/export?start=5000&end=15000&format=ubv"}}
        fetch = mock.Mock(return_value=None)
        with mock.patch.dict(os.environ, ENV), \
                mock.patch.object(ucp4_client, "STATE_DIR", "/tmp/x"), \
                mock.patch.object(ucp4_client, "_get_tagger", return_value=FakeTagger()), \
                mock.patch.object(ucp4_client.clip_fetch, "fetch_clip", fetch), \
                mock.patch.object(ucp4_client.requests, "post", post):
            ucp4_client._segment_video("10.0.0.5", "fcecdaaabbcc", body)
        url = fetch.call_args[0][0]
        self.assertIn("start=9000&end=11000", url)
        self.assertEqual(post.call_args.kwargs["json"], {"error": "Failed to download video"})


if __name__ == "__main__":
    unittest.main()
