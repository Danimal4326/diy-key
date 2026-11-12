# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for stage3_wss/lpr.py with a mocked plate reader (no fast-alpr needed)."""
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

import lpr  # noqa: E402


def _frame(bgr=(0, 0, 200)):
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    f[:] = bgr
    return f


def _meta(ts, tid, coord=(250, 250, 500, 500), obj="vehicle"):
    return {"ts": ts, "roi": {"coord": list(coord), "objectType": obj, "trackerId": tid}}


def _reader(table):
    """Reader returning table[call index] for successive calls."""
    calls = iter(table)
    return lambda crop: next(calls)


BOX = (10, 10, 110, 40)  # 100 px wide, over PLATE_MIN_PX


def check_descriptor(tc, d):
    """Mirror of Protect's descriptor zod schema (M/315148:26-38 all required; M/116837:33-49
    coord 4 finite numbers, zones/lines number arrays)."""
    tc.assertEqual(set(d) - {"attributes"}, {"boxColor", "confidenceLevel", "coord", "coord3d",
                                             "lines", "name", "objectType", "stationary",
                                             "tag", "trackerID", "zones"})
    tc.assertEqual(len(d["coord"]), 4)
    tc.assertTrue(all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in d["coord"]))
    tc.assertEqual(d["zones"], [])
    tc.assertEqual(d["lines"], [])
    tc.assertIsInstance(d["stationary"], bool)
    tc.assertIsInstance(d["trackerID"], int)


class BuildLprTest(unittest.TestCase):
    def test_schema(self):
        frames = {1000: _frame()}
        out = lpr.build_lpr("cam1", "ev1", frames, [_meta(1000, 7)], start_ms=1_700_000_000_000,
                            reader=_reader([[("abc123", 0.9, BOX)]]))
        self.assertIsNotNone(out)
        j, jpegs = out
        for k, t in (("cameraId", str), ("eventId", str), ("eventTracks", list),
                     ("lprAttrs", dict), ("lprSnapshots", list), ("inferMs", int),
                     ("preProcessMs", int), ("status", str), ("timeElapsedMs", int)):
            self.assertIsInstance(j[k], t, k)
        self.assertEqual(j["status"], "success")
        self.assertEqual(len(j["eventTracks"]), 1)
        trk, = j["eventTracks"]
        self.assertEqual(trk["timestamp"], 1000)
        d, = trk["descriptors"]
        check_descriptor(self, d)
        self.assertEqual(d["objectType"], "licensePlate")
        self.assertEqual(d["name"], "ABC123")
        self.assertEqual(d["trackerID"], 7 + lpr.PLATE_TRACKER_OFFSET)
        self.assertEqual(d["confidenceLevel"], 90)
        a = j["lprAttrs"]["7"]
        self.assertEqual(a["matchedName"], "ABC123")
        self.assertEqual(a["namesTopK"], ["ABC123"])
        self.assertEqual(a["objectType"], "vehicle")
        self.assertEqual(a["topKCandidate"], [{"name": "ABC123", "confidence": 90.0}])
        self.assertEqual(a["color"]["val"], "red")  # BGR (0,0,200) is red
        self.assertNotIn("vehicleType", a)          # objectType alone gives no type
        s = j["lprSnapshots"][0]
        self.assertEqual(set(s), {"clockBestMonotonic", "clockBestWall", "smartDetectHeatmap",
                                  "smartDetectSnapshot", "smartDetectSnapshotName",
                                  "smartDetectSnapshotType", "trackerID"})
        self.assertEqual(s["smartDetectSnapshotType"], "licensePlate")
        self.assertEqual(s["smartDetectSnapshot"], f"/tmp/cam1_{7 + lpr.PLATE_TRACKER_OFFSET}.jpg")
        self.assertEqual((s["smartDetectHeatmap"], s["smartDetectSnapshotName"]), ("", ""))
        self.assertEqual(s["clockBestMonotonic"], 1000)
        self.assertEqual(s["clockBestWall"], 1_700_000_001_000)
        self.assertEqual(s["trackerID"], 7 + lpr.PLATE_TRACKER_OFFSET)
        self.assertEqual(list(jpegs), [str(7 + lpr.PLATE_TRACKER_OFFSET)])
        self.assertTrue(jpegs[str(7 + lpr.PLATE_TRACKER_OFFSET)].startswith(b"\xff\xd8"))

    def test_best_read_per_tracker(self):
        frames = {1: _frame(), 2: _frame(), 3: _frame()}
        meta = [_meta(1, 5), _meta(2, 5), _meta(3, 9)]
        reads = [[("AB1234", 0.6, BOX)], [("AB1284", 0.8, BOX), ("XY", 0.99, BOX)],
                 [("ZZZ999", 0.7, BOX)]]
        j, jpegs = lpr.build_lpr("c", "e", frames, meta, reader=_reader(reads))
        self.assertEqual(j["lprAttrs"]["5"]["matchedName"], "AB1284")
        self.assertEqual(j["lprAttrs"]["5"]["namesTopK"], ["AB1284", "AB1234"])
        self.assertEqual(j["lprAttrs"]["9"]["matchedName"], "ZZZ999")
        snaps = {s["trackerID"]: s for s in j["lprSnapshots"]}
        off = lpr.PLATE_TRACKER_OFFSET
        self.assertEqual(set(snaps), {5 + off, 9 + off})
        self.assertEqual(snaps[5 + off]["clockBestWall"], 2)  # no start_ms: ts
        self.assertEqual(set(jpegs), {str(5 + off), str(9 + off)})

    def test_narrow_plate_and_non_vehicle_skipped(self):
        frames = {1: _frame(), 2: _frame()}
        meta = [_meta(1, 1), _meta(2, 2, obj="person")]
        out = lpr.build_lpr("c", "e", frames, meta,
                            reader=_reader([[("ABC123", 0.9, (0, 0, 30, 10))]]))
        self.assertIsNone(out)

    def test_vehicle_type_from_label(self):
        m = _meta(1, 3)
        m["roi"]["label"] = "truck"
        j, _ = lpr.build_lpr("c", "e", {1: _frame()}, [m],
                             reader=_reader([[("TRK001", 0.9, BOX)]]))
        self.assertEqual(j["lprAttrs"]["3"]["vehicleType"]["val"], "truck")

    def test_none_when_empty(self):
        self.assertIsNone(lpr.build_lpr("c", "e", {}, [], reader=lambda c: []))
        self.assertIsNone(lpr.build_lpr("c", "e", {1: _frame()}, [_meta(1, 1)],
                                        reader=lambda c: []))
        # missing frame
        self.assertIsNone(lpr.build_lpr("c", "e", {}, [_meta(1, 1)],
                                        reader=lambda c: [("ABC123", 0.9, BOX)]))

    def test_none_when_deps_missing_logs_once(self):
        with mock.patch.object(lpr, "lpr_available", return_value=False), \
                mock.patch.object(lpr, "_unavailable_logged", False), \
                self.assertLogs("diykey.lpr", "WARNING") as cm:
            self.assertIsNone(lpr.build_lpr("c", "e", {1: _frame()}, [_meta(1, 1)]))
            self.assertIsNone(lpr.build_lpr("c", "e", {1: _frame()}, [_meta(1, 1)]))
        self.assertEqual(len(cm.output), 1)


if __name__ == "__main__":
    unittest.main()
