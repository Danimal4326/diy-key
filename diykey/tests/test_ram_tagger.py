# SPDX-License-Identifier: AGPL-3.0-or-later
"""build_result: tagObjs from roiMeta (SW:2737-2760), and never searchSnapshots."""
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
from protect_labels import convert_ram_tags_to_labels  # noqa: E402


class FakeTagger:
    def tag_frame(self, img):
        return [{"tag": "person", "confScore": 0.9}]


class RamImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "f.jpg")
        cv2.imwrite(self.path, np.full((200, 400, 3), 120, np.uint8))

    def test_tagobjs_without_snapshots(self):
        roi = [{"coord": [100, 100, 300, 600], "objectType": "person", "trackerId": 5},
               {"coord": [0, 0, 0, 0], "objectType": "vehicle", "trackerId": 9}]
        meta = [{"ts": 1000, "roi": roi}, {"ts": 2000, "roi": roi[0]}]
        res = ram_tagger.build_result(FakeTagger(), "cam", "ev", {1000: self.path}, roi_meta=meta)
        km, = res["keyMomentsTags"]
        self.assertEqual(km["tagObjs"], roi)
        # Infer agent 0.0.24.4 never sets it; Protect makes a new detection per entry (live
        # 07:55 Backyard duplicate person, 16:18 Doorbell three tiles of a parked car).
        self.assertNotIn("searchSnapshots", km)

    def test_tracked_objects_use_thumbnail_tags(self):
        roi = {"coord": [100, 100, 300, 600], "objectType": "person", "trackerId": 5}
        objects = [{"trackerID": 5, "keyMomentMs": 1000, "img": cv2.imread(self.path), "hint": roi}]
        res = ram_tagger.build_result(FakeTagger(), "cam", "ev", {1000: self.path},
                                      objects=objects, roi_meta=[{"ts": 1000, "roi": [roi]}])
        self.assertEqual([t["trackerID"] for t in res["thumbnailTags"]], [5])

    def test_build_result_without_roi(self):
        res = ram_tagger.build_result(FakeTagger(), "cam", "ev", {1000: self.path})
        self.assertIsInstance(res, dict)
        self.assertNotIn("tagObjs", res["keyMomentsTags"][0])


class FakeSession:
    """Stands in for the RAM++ onnxruntime session: fixed probabilities over 5 tags."""
    def __init__(self, probs):
        self.probs, self.seen = np.array(probs, np.float32), []

    def get_inputs(self):
        return [type("I", (), {"name": "image"})()]

    def run(self, _, feed):
        self.seen.append(feed["image"].shape)
        return [self.probs[None]]


def make_plus(tmp, probs, vocab, **kw):
    plus = ram_tagger.RamPlus(os.path.join(tmp, "ram_plus.onnx"), vocab=vocab, **kw)
    plus._session = FakeSession(probs)
    plus.names = ["man", "sofa", "shirt", "sedan", "glasses"]
    plus.thresholds = np.array([0.7, 0.7, 0.5, 0.6, 0.9], np.float32)
    plus.keep = np.array([n in vocab for n in plus.names])
    return plus


class RamPlusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_per_class_thresholds_and_vocab(self):
        # sofa passes its threshold but is not an enabled Protect tag; glasses (0.8 < 0.9) and
        # sedan (0.55 < 0.6) fall short of their own thresholds; shirt passes at 0.55 because its threshold is 0.5.
        plus = make_plus(self.tmp.name, [0.8, 0.9, 0.55, 0.55, 0.8],
                         {"man", "shirt", "sedan", "glasses"})
        tags = plus.tags(np.zeros((50, 30, 3), np.uint8))
        self.assertEqual(list(tags), ["man", "shirt"])
        self.assertEqual(plus._session.seen, [(1, 3, 384, 384)])

    def test_max_tags_keeps_best(self):
        plus = make_plus(self.tmp.name, [0.8, 0.9, 0.95, 0.99, 0.97],
                         {"man", "shirt", "sedan", "glasses"}, max_tags=2)
        self.assertEqual(list(plus.tags(np.zeros((8, 8, 3), np.uint8))), ["sedan", "glasses"])

    def test_failure_disables_and_returns_empty(self):
        plus = ram_tagger.RamPlus(os.path.join(self.tmp.name, "missing.onnx"))
        self.assertEqual(plus.tags(np.zeros((8, 8, 3), np.uint8)), {})
        self.assertTrue(plus._failed)

    def test_filter_by_object_type(self):
        tags = {"man": 0.9, "sedan": 0.8, "shirt": 0.7}
        self.assertEqual(set(ram_tagger.plus_filter(tags, "person")), {"man", "shirt"})
        self.assertEqual(set(ram_tagger.plus_filter(tags, "vehicle")), {"sedan", "shirt"})
        self.assertEqual(set(ram_tagger.plus_filter(tags, "animal")), {"shirt"})
        self.assertEqual(set(ram_tagger.plus_filter(tags, None)), set(tags))

    def test_crop_keeps_only_label_tags(self):
        tags = {"man": 0.9, "oven": 0.86, "food": 0.8, "black shirt": 0.7, "glasses": 0.7,
                "person": 0.8, "tray": 0.6}
        self.assertEqual(set(ram_tagger.plus_filter(tags, "person")),
                         {"man", "black shirt", "glasses", "person"})
        self.assertEqual(set(ram_tagger.plus_filter(tags, None)), set(tags))  # whole frame
        for t in ("man", "oven", "blue sedan", "bus", "hat", "tray", "short", "grey coat"):
            self.assertEqual(ram_tagger.label_tag(t),
                             bool(convert_ram_tags_to_labels("vehicle", [t]) or
                                  convert_ram_tags_to_labels("person", [t])), t)

    def test_unset_keeps_current_behaviour(self):
        os.environ.pop("DIYKEY_RAM_PLUS_MODEL", None)
        self.assertIsNone(getattr(FakeTagger(), "plus", None))
        res = ram_tagger.tag_crop(FakeTagger(), np.zeros((20, 20, 3), np.uint8),
                                  {"objectType": "person", "confidence": 90})
        self.assertEqual([t["tag"] for t in res], ["person"])


class PlusTagger(FakeTagger):
    def __init__(self, plus):
        self.plus = plus

    def tag_frame(self, img, plus=True):
        out = [{"tag": "person", "confScore": 0.9}, {"tag": "car", "confScore": 0.8}]
        if plus:
            out.append({"tag": "sofa", "confScore": 0.6})
        return out


class PlusMergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plus = make_plus(self.tmp.name, [0.8, 0.9, 0.9, 0.9, 0.95],
                              {"man", "shirt", "sedan", "glasses"})
        self.img = np.full((60, 40, 3), 90, np.uint8)

    def test_person_crop_merges_plus_tags(self):
        tags = ram_tagger.tag_crop(PlusTagger(self.plus), self.img,
                                   {"objectType": "person", "confidence": 90})
        got = {t["tag"]: t["confScore"] for t in tags}
        self.assertEqual(set(got), {"person", "man", "shirt", "glasses"})  # no sedan, no car
        self.assertEqual(got["glasses"], 0.95)
        self.assertEqual([t["tag"] for t in tags][0], "glasses")  # sorted by score

    def test_vehicle_crop_drops_person_tags_and_defers_to_camera(self):
        tagger = PlusTagger(self.plus)
        tags = ram_tagger.tag_crop(tagger, self.img, {"objectType": "vehicle"})
        self.assertEqual({t["tag"] for t in tags}, {"car", "sedan", "shirt", "glasses"})
        hint = {"objectType": "vehicle",
                "attributes": {"vehicleType": {"val": "suv", "confidence": 80}}}
        tags = ram_tagger.tag_crop(tagger, self.img, hint)
        self.assertEqual({t["tag"] for t in tags}, {"suv", "shirt", "glasses"})

    def test_whole_frame_and_thumbnails_get_labels(self):
        path = os.path.join(self.tmp.name, "f.jpg")
        cv2.imwrite(path, self.img)
        tagger = PlusTagger(self.plus)
        tagger.plus = self.plus
        # The whole-frame path: PlusTagger.tag_frame(plus=True) stands for RamTagger's merge.
        res = ram_tagger.build_result(tagger, "cam", "ev", {1000: path},
                                      objects=[{"trackerID": 3, "keyMomentMs": 1000,
                                                "img": self.img,
                                                "hint": {"objectType": "person", "confidence": 90}}])
        thumb, = res["thumbnailTags"]
        names = [t["tag"] for t in thumb["tags"]]
        self.assertEqual(sorted(convert_ram_tags_to_labels("person", names)),
                         ["accessory:glasses", "clothingType:shirt", "gender:male"])
        self.assertIn("sofa", [t["tag"] for t in res["keyMomentsTags"][0]["tags"]])

    def test_labels_port(self):
        self.assertEqual(convert_ram_tags_to_labels("vehicle", ["blue sedan", "bus"]),
                         ["vehicleType:car", "color:blue", "vehicleType:bus"])
        self.assertEqual(convert_ram_tags_to_labels("person", ["black shirt", "woman", "glass"]),
                         ["upperClothingColor:black", "clothingType:shirt", "gender:female",
                          "accessory:glasses"])


if __name__ == "__main__":
    unittest.main()
