# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for stage3_wss/scene_memory.py: real frames on disk, a scripted detector, and a
fingerprint that is the crop's mean colour, so "same look" means "same colour" (no models load).
The scenario is the one seen live on 2026-11-08 Driveway: a car parked in frame while others pass."""
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import scene_memory as sm  # noqa: E402

SIZE = 400
T0 = 1_790_000_000_000
MIN = 60_000
DAY = 1440 * MIN
BG = (30, 160, 30)              # colour, so frames count as daytime
RED, BLUE, WHITE = (0, 0, 255), (255, 0, 0), (250, 250, 180)
SPOT = [100, 500, 300, 300]     # where the car parks
ROAD = [600, 100, 200, 150]     # away from the spot
NEAR = [150, 450, 300, 300]


def fake_embed(kind, crop):
    return [float(v) for v in crop.reshape(-1, 3).mean(0)]


def obj(label, coord, colour=RED, detected=True):
    return {"label": label, "coord": coord, "colour": colour, "detected": detected}


def trig(ms, coord, otype="person"):
    return {"ts": ms, "coord": coord, "objectType": otype}


class SceneTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        env = mock.patch.dict(os.environ, {"DIYKEY_SCENE_MEMORY": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.n = 0

    def _frame(self, ms, objects, night):
        img = np.full((SIZE, SIZE, 3), 110 if night else 0, np.uint8)
        if not night:
            img[:] = BG
        for o in objects:
            x, y, w, h = (int(v * SIZE / 1000) for v in o["coord"])
            img[y:y + h, x:x + w] = 180 if night else o["colour"]
        path = os.path.join(self.d, f"f{ms}.png")
        cv2.imwrite(path, img)
        return path

    def event(self, ms, objects=(), triggers=(), frames=None, night=False, camera="cam",
              embed=fake_embed, namer=None):
        """One event. objects is what's in the picture; frames, if given, is a list of those per
        frame (for things that move during the event)."""
        self.n += 1
        lists = frames if frames is not None else [list(objects)]
        paths, script = {}, []
        for i, objs in enumerate(lists):
            at = ms + i
            paths[at] = self._frame(at, objs, night)
            script.append([(o["label"], 0.9, tuple(v * SIZE / 1000 for v in o["coord"]))
                           for o in objs if o["detected"]])
        tagger = mock.Mock()
        tagger.detect.side_effect = script
        return sm.observe(self.d, tagger, camera, f"ev{self.n}", paths, list(triggers),
                          embed=embed, namer=namer or (lambda t, k, c: []))

    def entries(self, camera="cam"):
        path = sm._path(self.d, camera)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)["entries"]

    def state(self, camera="cam"):
        return [e["state"] for e in self.entries(camera)]

    def park(self, label="car", coord=SPOT, colour=RED, namer=None, gap=16):
        """Three events over 32 min, with a person walking past on the road each time."""
        scene = None
        for i in range(3):
            at = T0 + i * gap * MIN
            scene = self.event(at, [obj(label, coord, colour)], [trig(at, ROAD)], namer=namer)
        return scene

    def later(self, n=1):
        return T0 + (60 + n) * MIN


class TestPromotion(SceneTest):
    def test_vehicle_after_three_events_over_thirty_minutes(self):
        self.park()
        self.assertEqual(self.state(), ["stationary"])

    def test_animal_rule_and_summary(self):
        scene = None
        for i in range(3):
            at = T0 + i * 6 * MIN
            scene = self.event(at, [obj("dog", SPOT)], [trig(at, ROAD)], namer=lambda *a: [])
        self.assertEqual(self.state(), ["stationary"])
        self.assertEqual(scene.still, ["The dog is still resting."])
        self.assertIn("still resting", sm.summary_text("A person walks.", scene))

    def test_person_rule_and_summary(self):
        scene = None
        for i in range(3):
            at = T0 + i * 6 * MIN
            scene = self.event(at, [obj("person", SPOT)], [trig(at, ROAD, "vehicle")])
        self.assertEqual(self.state(), ["stationary"])
        self.assertEqual(scene.still, ["The person is still there."])

    def test_looser_box_match_for_people_weights_appearance(self):
        for i, coord in enumerate((SPOT, NEAR, SPOT)):
            at = T0 + i * 6 * MIN
            self.event(at, [obj("person", coord)], [trig(at, ROAD, "vehicle")])
        self.assertEqual(self.state(), ["stationary"])

    def test_a_vehicle_needs_the_tight_box(self):
        for i, coord in enumerate((SPOT, [300, 500, 300, 300], SPOT)):
            at = T0 + i * 16 * MIN
            self.event(at, [obj("car", coord)], [trig(at, ROAD)])
        self.assertNotIn("stationary", self.state())

    def test_animal_expires_after_an_hour_unseen(self):
        for i in range(3):
            at = T0 + i * 6 * MIN
            self.event(at, [obj("dog", SPOT)], [trig(at, ROAD)])
        at = T0 + 12 * MIN + 61 * MIN
        scene = self.event(at, [obj("dog", SPOT)], [trig(at, ROAD)])
        self.assertEqual(self.state(), ["candidate"])
        self.assertEqual(scene.still, [])

    def test_person_expires_after_fifteen_minutes_unseen(self):
        for i in range(3):
            at = T0 + i * 6 * MIN
            self.event(at, [obj("person", SPOT)], [trig(at, ROAD, "vehicle")])
        at = T0 + 12 * MIN + 16 * MIN
        scene = self.event(at, [obj("person", SPOT)], [trig(at, ROAD, "vehicle")])
        self.assertEqual(self.state(), ["candidate"])
        self.assertEqual(scene.still, [])

    def test_settings_override_the_rules(self):
        with mock.patch.dict(os.environ, {"DIYKEY_SCENE_VEHICLE_EVENTS": "2",
                                          "DIYKEY_SCENE_VEHICLE_MINUTES": "1"}):
            for i in range(2):
                at = T0 + i * 2 * MIN
                self.event(at, [obj("car", SPOT)], [trig(at, ROAD)])
        self.assertEqual(self.state(), ["stationary"])


class TestTimeSpan(SceneTest):
    def test_three_quick_events_are_not_enough(self):
        for i in range(3):
            at = T0 + i * MIN
            self.event(at, [obj("car", SPOT)], [trig(at, ROAD)])
        self.assertEqual(self.state(), ["candidate"])

    def test_promoted_once_the_span_is_reached(self):
        for minute in (0, 1, 2, 31):
            at = T0 + minute * MIN
            self.event(at, [obj("car", SPOT)], [trig(at, ROAD)])
        self.assertEqual(self.state(), ["stationary"])

    def test_counts_once_per_event(self):
        self.event(T0, [obj("car", SPOT)], [trig(T0, ROAD)])
        self.event(T0 + 16 * MIN, frames=[[obj("car", SPOT)]] * 3,
                   triggers=[trig(T0 + 16 * MIN, ROAD)])
        self.assertEqual(self.entries()[0]["count"], 2)

    def test_a_replayed_event_is_not_counted_twice(self):
        path = self._frame(T0, [obj("car", SPOT)], False)
        tagger = mock.Mock()
        tagger.detect.return_value = [("car", 0.9, (40, 200, 120, 120))]
        for _ in range(2):
            sm.observe(self.d, tagger, "cam", "same", {T0: path}, [trig(T0, ROAD)],
                       embed=fake_embed, namer=lambda *a: [])
        self.assertEqual(self.entries()[0]["count"], 1)


class TestPassingCars(SceneTest):
    def test_the_events_own_trigger_is_not_learned(self):
        for i in range(4):
            at = T0 + i * 16 * MIN
            self.event(at, [obj("car", ROAD)], [trig(at, ROAD, "vehicle")])
        self.assertEqual(self.entries(), [])

    def test_a_car_moving_during_the_event_is_not_learned(self):
        boxes = ([100, 100, 200, 150], [400, 100, 200, 150], [700, 100, 200, 150])
        for i in range(4):
            at = T0 + i * 16 * MIN
            self.event(at, frames=[[obj("car", b)] for b in boxes], triggers=[trig(at, [0, 0, 5, 5])])
        self.assertEqual(self.entries(), [])

    def test_different_cars_on_the_same_stretch_are_not_learned(self):
        for i, colour in enumerate((RED, BLUE, WHITE, RED, BLUE)):
            at = T0 + i * 16 * MIN
            self.event(at, [obj("car", ROAD, colour)], [trig(at, [0, 0, 5, 5])])
        self.assertNotIn("stationary", self.state())

    def test_one_off_cars_are_forgotten(self):
        self.event(T0, [obj("car", ROAD)], [trig(T0, [0, 0, 5, 5])])
        for i in (1, 2):
            self.event(T0 + i * MIN, [], [trig(T0 + i * MIN, [0, 0, 5, 5])])
        self.assertEqual(self.entries(), [])

    def test_a_passing_car_does_not_disturb_the_parked_one(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT), obj("car", ROAD, BLUE)],
                           [trig(at, ROAD, "vehicle")])
        self.assertEqual(scene.changes, [])
        self.assertEqual(self.state(), ["stationary"])


class TestPeopleNeverHidden(SceneTest):
    def test_tags_stay_after_hours(self):
        scene, at = None, T0
        for i in range(36):  # six hours, every ten minutes
            at = T0 + i * 10 * MIN
            scene = self.event(at, [obj("person", SPOT)], [trig(at, ROAD, "vehicle")])
        self.assertEqual(self.state(), ["stationary"])
        self.assertEqual(scene.still, ["The person is still there."])
        self.assertEqual(scene.ignore, [])
        self.assertEqual(scene.hide, set())
        result = sm.filter_result({"description": "x", "keyMomentsTags": [
            {"keyMomentMs": at, "tags": [{"tag": "person", "confScore": 0.9}]}]}, scene)
        self.assertEqual(result["keyMomentsTags"][0]["tags"][0]["tag"], "person")

    def test_a_person_fixture_keeps_its_tags(self):
        scene = self._statue()
        result = sm.filter_result({"description": "x", "keyMomentsTags": [
            {"keyMomentMs": 1, "tags": [{"tag": "person", "confScore": 0.9}]}]}, scene)
        self.assertEqual(len(result["keyMomentsTags"][0]["tags"]), 1)

    def _statue(self):
        scene = None
        for day in range(9):
            at = T0 + day * DAY
            scene = self.event(at, [obj("person", SPOT)], [trig(at, ROAD, "vehicle")])
        self.assertEqual(self.state(), ["fixture"])
        return scene


class TestFixture(SceneTest):
    def daily(self, days, label="person", colour=RED):
        scene = None
        for day in range(days):
            at = T0 + day * DAY
            scene = self.event(at, [obj(label, SPOT, colour)],
                               [trig(at, ROAD, "vehicle")])
        return scene

    def test_a_statue_becomes_a_fixture_after_seven_days(self):
        scene = self.daily(8)
        self.assertEqual(self.state(), ["fixture"])
        self.assertEqual(scene.still, [])
        self.assertIn("motionless figure", sm.ignore_context(scene))

    def test_not_before_seven_days(self):
        self.daily(7)
        self.assertNotEqual(self.state(), ["fixture"])

    def test_works_for_any_object(self):
        self.daily(8, "dog")
        self.assertEqual(self.state(), ["fixture"])

    def test_drops_out_the_moment_its_look_changes(self):
        self.daily(8)
        at = T0 + 9 * DAY
        scene = self.event(at, [obj("person", SPOT, BLUE)], [trig(at, ROAD, "vehicle")])
        self.assertNotIn("fixture", self.state())
        self.assertEqual(scene.changes, [])

    def test_absence_restarts_the_clock(self):
        self.daily(4)
        for day in (4, 5):
            at = T0 + day * DAY
            self.event(at, [], [trig(at, ROAD, "vehicle")])
        self.assertEqual(self.entries(), [])
        for day in range(6, 10):
            at = T0 + day * DAY
            self.event(at, [obj("person", SPOT)], [trig(at, ROAD, "vehicle")])
        self.assertNotEqual(self.state(), ["fixture"])


class TestTagRemoval(SceneTest):
    TAGS = ["car", "sedan", "suv", "white car", "vehicle", "car park", "tree", "person", "dog"]

    def result(self):
        tags = [{"tag": t, "confScore": 0.9 - i / 100} for i, t in enumerate(self.TAGS)]
        return {"cameraId": "cam", "eventId": "e", "description": "Car detected.",
                "keyMomentsTags": [{"keyMomentMs": 1, "tags": tags, "tagObjs": [{"coord": [1]}]}],
                "thumbnailTags": [{"trackerID": 7, "keyMomentMs": 1,
                                   "tags": [{"tag": "car", "confScore": 0.5}]}]}

    def test_removed_when_the_trigger_is_something_else(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT)], [trig(at, ROAD, "person")])
        self.assertEqual(scene.hide, {"vehicle"})
        out = sm.filter_result(self.result(), scene)
        names = [t["tag"] for t in out["keyMomentsTags"][0]["tags"]]
        self.assertEqual(names, ["car park", "tree", "person", "dog"])

    def test_tracked_objects_and_format_are_untouched(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT)], [trig(at, ROAD, "person")])
        before = self.result()
        out = sm.filter_result(self.result(), scene)
        self.assertEqual(out["thumbnailTags"], before["thumbnailTags"])
        self.assertEqual(out["keyMomentsTags"][0]["tagObjs"], [{"coord": [1]}])
        self.assertEqual(set(out), set(before))
        self.assertEqual(set(out["keyMomentsTags"][0]), set(before["keyMomentsTags"][0]))
        self.assertNotIn("searchSnapshots", out["keyMomentsTags"][0])
        self.assertNotIn("car", out["description"].lower().replace("car park", ""))

    def test_kept_when_the_trigger_is_a_vehicle(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT)], [trig(at, ROAD, "vehicle")])
        self.assertEqual(scene.hide, set())

    def test_kept_when_the_trigger_is_unknown(self):
        self.park()
        at = self.later()
        self.assertEqual(self.event(at, [obj("car", SPOT)], []).hide, set())
        at += MIN
        self.assertEqual(self.event(at, [obj("car", SPOT)], [trig(at, ROAD, None)]).hide, set())

    def test_kept_when_another_vehicle_is_not_stationary(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT), obj("car", ROAD, BLUE)],
                           [trig(at, [0, 0, 5, 5], "person")])
        self.assertEqual(scene.hide, set())

    def test_animals_too(self):
        for i in range(3):
            at = T0 + i * 6 * MIN
            self.event(at, [obj("dog", SPOT)], [trig(at, ROAD)])
        at = T0 + 14 * MIN
        scene = self.event(at, [obj("dog", SPOT)], [trig(at, ROAD)])
        self.assertEqual(scene.hide, {"animal"})
        out = sm.filter_result(self.result(), scene)
        names = [t["tag"] for t in out["keyMomentsTags"][0]["tags"]]
        self.assertNotIn("dog", names)
        self.assertIn("car", names)

    def test_nothing_changes_without_a_scene(self):
        self.assertEqual(sm.filter_result(self.result(), None), self.result())


class TestSummaryLines(SceneTest):
    def test_ignore_line_names_the_parked_car(self):
        self.park(namer=lambda t, k, c: [{"tag": "red"}, {"tag": "sedan"}])
        at = self.later()
        scene = self.event(at, [obj("car", SPOT)], [trig(at, ROAD)])
        self.assertEqual(sm.ignore_context(scene),
                         "Ignore the parked red sedan. Do not mention them.")

    def test_no_line_without_anything_to_ignore(self):
        self.assertIsNone(sm.ignore_context(sm.EventScene()))
        self.assertIsNone(sm.ignore_context(None))


class TestMoveDetection(SceneTest):
    NAMER = staticmethod(lambda t, k, c: [{"tag": "red"}, {"tag": "sedan"}])

    def test_the_trigger_is_the_parked_car_somewhere_else(self):
        self.park(namer=self.NAMER)
        at = self.later()
        scene = self.event(at, [obj("car", ROAD)], [trig(at, ROAD, "vehicle")])
        self.assertEqual(scene.changes, ["The parked red sedan has moved."])
        self.assertEqual(self.entries(), [])

    def test_a_lookalike_passing_while_it_is_still_parked(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT), obj("car", ROAD)], [trig(at, ROAD, "vehicle")])
        self.assertEqual(scene.changes, [])
        self.assertEqual(self.state(), ["stationary"])

    def test_missing_from_two_events_with_its_spot_visible(self):
        self.park()
        first = self.event(self.later(1), [], [trig(self.later(1), ROAD)])
        self.assertEqual(first.changes, [])
        self.assertEqual(self.state(), ["stationary"])
        second = self.event(self.later(2), [], [trig(self.later(2), ROAD)])
        self.assertEqual(second.changes, ["The parked car has moved."])
        self.assertEqual(self.entries(), [])

    def test_seen_again_resets_the_missing_count(self):
        self.park()
        self.event(self.later(1), [], [trig(self.later(1), ROAD)])
        self.event(self.later(2), [obj("car", SPOT)], [trig(self.later(2), ROAD)])
        scene = self.event(self.later(3), [], [trig(self.later(3), ROAD)])
        self.assertEqual(scene.changes, [])

    def test_a_blocked_spot_is_not_missing(self):
        self.park()
        for n in (1, 2, 3):
            at = self.later(n)
            scene = self.event(at, [], [trig(at, SPOT)])
            self.assertEqual(scene.changes, [])
        self.assertEqual(self.state(), ["stationary"])

    def test_a_detector_miss_is_not_missing(self):
        self.park()
        for n in (1, 2, 3):
            at = self.later(n)
            scene = self.event(at, [obj("car", SPOT, detected=False)], [trig(at, ROAD)])
            self.assertEqual(scene.changes, [])
            self.assertEqual(scene.hide, {"vehicle"})
        self.assertEqual(self.state(), ["stationary"])

    def test_a_different_looking_object_in_its_spot(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT, BLUE)], [trig(at, ROAD)])
        self.assertEqual(scene.changes, ["The parked car has moved."])
        self.assertEqual(self.state(), ["candidate"])

    def test_something_passing_over_the_spot_is_not_a_replacement(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT, BLUE)], [trig(at, SPOT, "vehicle")])
        self.assertEqual(scene.changes, [])

    def test_a_person_gets_the_same_treatment(self):
        for i in range(3):
            at = T0 + i * 6 * MIN
            self.event(at, [obj("person", SPOT)], [trig(at, ROAD, "vehicle")])
        for n, want in ((13, []), (14, ["The person who was seated has left."])):
            at = T0 + n * MIN
            self.assertEqual(self.event(at, [], [trig(at, ROAD, "vehicle")]).changes, want)

    def test_an_animal_gets_the_same_treatment(self):
        for i in range(3):
            at = T0 + i * 6 * MIN
            self.event(at, [obj("dog", SPOT)], [trig(at, ROAD)])
        for n, want in ((13, []), (14, ["The dog that was resting has moved."])):
            at = T0 + n * MIN
            self.assertEqual(self.event(at, [], [trig(at, ROAD)]).changes, want)

    def test_the_sentence_goes_first_in_summary_and_description(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT, BLUE)], [trig(at, ROAD)])
        self.assertEqual(sm.summary_text("A man walks.", scene),
                         "The parked car has moved. A man walks.")
        out = sm.filter_result({"description": "Person detected."}, scene)
        self.assertEqual(out["description"], "The parked car has moved. Person detected.")

    def test_a_candidate_is_dropped_without_a_report(self):
        self.event(T0, [obj("car", SPOT)], [trig(T0, ROAD)])
        for n in (1, 2):
            scene = self.event(T0 + n * MIN, [], [trig(T0 + n * MIN, ROAD)])
            self.assertEqual(scene.changes, [])

    def test_night_is_not_compared_with_day(self):
        self.park()
        at = self.later()
        scene = self.event(at, [obj("car", SPOT)], [trig(at, ROAD)], night=True)
        self.assertEqual(scene.changes, [])
        entry = self.entries()[0]
        self.assertEqual(entry["state"], "stationary")
        self.assertEqual(set(entry["fp"]), {"day", "night"})

    def test_no_fingerprint_for_this_light_is_not_missing(self):
        self.park()
        for n in (1, 2, 3):
            at = self.later(n)
            self.assertEqual(self.event(at, [], [trig(at, ROAD)], night=True).changes, [])
        self.assertEqual(self.state(), ["stationary"])


class TestRelearning(SceneTest):
    def test_learned_again_in_its_new_spot(self):
        self.park()
        at = self.later()
        self.event(at, [obj("car", SPOT, BLUE)], [trig(at, ROAD)])
        for i in range(1, 4):
            t = at + i * 16 * MIN
            scene = self.event(t, [obj("car", SPOT, BLUE)], [trig(t, ROAD)])
        self.assertEqual(self.state(), ["stationary"])
        self.assertEqual(scene.hide, {"vehicle"})

    def test_a_moved_car_starts_over(self):
        self.park()
        at = self.later()
        self.event(at, [obj("car", ROAD)], [trig(at, ROAD, "vehicle")])
        t = at + MIN
        self.event(t, [obj("car", ROAD)], [trig(t, [0, 0, 5, 5])])
        self.assertEqual(self.state(), ["candidate"])
        self.assertEqual(self.entries()[0]["count"], 1)


class TestPersistence(SceneTest):
    def test_survives_a_restart(self):
        global sm
        self.park()
        sm = importlib.reload(sm)
        at = self.later()
        scene = self.event(at, [obj("car", SPOT)], [trig(at, ROAD)])
        self.assertEqual(scene.hide, {"vehicle"})

    def test_each_camera_has_its_own_file(self):
        self.event(T0, [obj("car", SPOT)], [trig(T0, ROAD)], camera="a")
        self.event(T0, [obj("car", ROAD)], [trig(T0, SPOT)], camera="b/../c")
        names = sorted(os.listdir(os.path.join(self.d, sm.SUBDIR)))
        self.assertEqual(names, ["a.json", "b____c.json"])  # no path tricks in a camera id

    def test_written_atomically_as_json(self):
        self.event(T0, [obj("car", SPOT)], [trig(T0, ROAD)])
        files = os.listdir(os.path.join(self.d, sm.SUBDIR))
        self.assertEqual(files, ["cam.json"])
        entry = self.entries()[0]
        self.assertEqual(entry["kind"], "vehicle")
        self.assertEqual(entry["box"], SPOT)
        self.assertEqual(entry["count"], 1)

    def test_a_corrupt_file_starts_again(self):
        os.makedirs(os.path.join(self.d, sm.SUBDIR))
        with open(sm._path(self.d, "cam"), "w") as f:
            f.write("{not json")
        with self.assertLogs("diykey.scene", "WARNING"):
            scene = self.event(T0, [obj("car", SPOT)], [trig(T0, ROAD)])
        self.assertIsNotNone(scene)
        self.assertEqual(self.state(), ["candidate"])

    def test_a_parked_car_outlasts_a_long_quiet_spell(self):
        self.park()
        at = T0 + 30 * DAY
        scene = self.event(at, [obj("car", SPOT)], [trig(at, ROAD)])
        self.assertEqual(scene.hide, {"vehicle"})


class TestFailureIsolation(SceneTest):
    def test_off_by_default(self):
        tagger = mock.Mock()
        with mock.patch.dict(os.environ, {"DIYKEY_SCENE_MEMORY": ""}):
            self.assertIsNone(sm.observe(self.d, tagger, "c", "e", {1: "x"}, []))
        tagger.detect.assert_not_called()

    def test_an_embedding_failure_is_logged_and_contained(self):
        def boom(kind, crop):
            raise RuntimeError("model gone")
        with self.assertLogs("diykey.scene", "WARNING") as cm:
            self.assertIsNone(self.event(T0, [obj("car", SPOT)], [trig(T0, ROAD)], embed=boom))
        self.assertIn("model gone", cm.output[0])

    def test_a_detector_failure_is_contained(self):
        path = self._frame(T0, [], False)
        tagger = mock.Mock()
        tagger.detect.side_effect = RuntimeError("no")
        with self.assertLogs("diykey.scene", "WARNING"):
            self.assertIsNone(sm.observe(self.d, tagger, "c", "e", {T0: path}, []))

    def test_an_unwritable_state_dir_is_contained(self):
        path = self._frame(T0, [], False)
        tagger = mock.Mock()
        tagger.detect.return_value = []
        blocked = os.path.join(self.d, "file")
        open(blocked, "w").close()
        with self.assertLogs("diykey.scene", "WARNING"):
            self.assertIsNone(sm.observe(blocked, tagger, "c", "e", {T0: path}, []))

    def test_no_fingerprint_model_means_nothing_learned(self):
        scene = self.event(T0, [obj("car", SPOT)], [trig(T0, ROAD)], embed=lambda k, c: None)
        self.assertEqual(self.entries(), [])
        self.assertEqual(scene.hide, set())

    def test_a_naming_failure_still_promotes(self):
        def boom(*a):
            raise RuntimeError("no tags")
        self.park(namer=boom)
        self.assertEqual(self.state(), ["stationary"])
        self.assertEqual(self.entries()[0]["name"], "car")

    def test_a_bad_setting_falls_back(self):
        with mock.patch.dict(os.environ, {"DIYKEY_SCENE_VEHICLE_EVENTS": "many"}):
            self.park()
        self.assertEqual(self.state(), ["stationary"])

    def test_a_bad_result_is_returned_unchanged(self):
        scene = sm.EventScene()
        scene.hide = {"vehicle"}
        bad = {"description": "x", "keyMomentsTags": [{"tags": [{"no": "tag"}]}]}
        with self.assertLogs("diykey.scene", "WARNING"):
            self.assertIs(sm.filter_result(bad, scene), bad)

    def test_summary_helpers_tolerate_nothing(self):
        self.assertEqual(sm.summary_text("A car.", None), "A car.")
        self.assertIsNone(sm.summary_text(None, sm.EventScene()))


class TestLegacyKinds(SceneTest):
    def test_body_type_aliases_are_known(self):
        for otype in ("vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "suv", "van",
                      "motorbike", "bike", "pickup truck"):
            self.assertEqual(sm._kind(otype), sm.VEHICLE, otype)
        for otype in ("animal", "dog", "cat", "bird"):
            self.assertEqual(sm._kind(otype), sm.ANIMAL, otype)
        self.assertEqual(sm._kind("person"), sm.PERSON)

    def test_hide_words_cover_the_tagger_vocabulary(self):
        self.assertTrue({"deer", "raccoon", "squirrel", "wildlife"}
                        <= sm._HIDE_WORDS[sm.ANIMAL])
        self.assertTrue({"ambulance", "scooter", "suv", "van"}
                        <= sm._HIDE_WORDS[sm.VEHICLE])


class TestEventOrder(SceneTest):
    def test_an_older_event_does_not_rewind_the_clock(self):
        self.event(self.later(60), [obj("car", SPOT)], [trig(self.later(60), ROAD)])
        before = self.entries()
        self.assertIsNone(self.event(T0, [obj("car", SPOT)], [trig(T0, ROAD)]))
        self.assertEqual(self.entries(), before)


if __name__ == "__main__":
    unittest.main()
