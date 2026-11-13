# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for clip_fetch.py: URL construction and clip pruning."""
import os
import shutil
import sys
import tempfile
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import clip_fetch  # noqa: E402


class TestClipFetch(unittest.TestCase):
    def test_mp4_url_relative(self):
        result = clip_fetch.mp4_url('198.51.100.1', 7444,
            '/internal/aiprocessors/video/export?camera=c&format=ubv&mute=true')
        self.assertEqual(
            result,
            'https://198.51.100.1:7444/internal/aiprocessors/video/export?camera=c&format=mp4&mute=true'
        )

    def test_mp4_url_absolute_kept(self):
        input_url = 'https://10.0.0.1:7444/x?format=ubv'
        result = clip_fetch.mp4_url('10.0.0.1', 7444, input_url)
        self.assertEqual(result, 'https://10.0.0.1:7444/x?format=mp4')

    def test_mp4_url_adds_format(self):
        result1 = clip_fetch.mp4_url('h', 7444, '/x?a=1')
        self.assertEqual(result1, 'https://h:7444/x?a=1&format=mp4')
        result2 = clip_fetch.mp4_url('h', 7444, '/x')
        self.assertEqual(result2, 'https://h:7444/x?format=mp4')

    def test_prune_keeps_newest(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)

        # Create e0.mp4..e4.mp4 and a folder named e0
        files = [
            os.path.join(d, "e0.mp4"),
            os.path.join(d, "e1.mp4"),
            os.path.join(d, "e2.mp4"),
            os.path.join(d, "e3.mp4"),
            os.path.join(d, "e4.mp4"),
        ]
        for f in files:
            with open(f, "w") as fh:
                fh.write("x")

        # Create e0 folder with a file inside
        e0_dir = os.path.join(d, "e0")
        os.makedirs(e0_dir)
        with open(os.path.join(e0_dir, "inside"), "w") as fh:
            fh.write("x")

        # Set mtimes so e4 is newest, e3 second, ..., e0 oldest
        for i, f in enumerate(files):
            os.utime(f, (1000 + i, 1000 + i))

        # Also set e0 dir mtime to be before e0.mp4 (so it gets pruned with e0.mp4)
        os.utime(e0_dir, (999, 999))

        clip_fetch.prune(d, keep=2)

        remaining_mp4 = set(os.listdir(d))
        self.assertEqual(remaining_mp4, {"e3.mp4", "e4.mp4"})
        self.assertFalse(os.path.exists(e0_dir))

    def test_clip_start_prefers_header(self):
        url = "/x?start=1000&end=31000&format=mp4"
        self.assertEqual(clip_fetch.clip_start(url, "500", "c", probe=lambda p: 35000),
                         (500, "x-start-timestamp"))

    def test_clip_start_from_duration(self):
        # Live 07:54 Backyard: 31.3 s window, 35.5 s clip -> the clip began ~4 s early.
        url = "/x?start=1794142489480&end=1794142520817&format=mp4"
        self.assertEqual(clip_fetch.clip_start(url, None, "c", probe=lambda p: 35467),
                         (1794142520817 - 35467, "end - duration"))
        self.assertEqual(clip_fetch.clip_start(url, "0", "c", probe=lambda p: 20000),
                         (1794142489480, "request start"))

    def test_extract_frames_seeks_from_saved_start(self):
        import subprocess
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            clip = os.path.join(d, "e.mp4")
            open(clip, "w").close()
            with open(clip + ".start", "w") as f:
                f.write("6000")
            with mock.patch.object(subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess([], 1, "", "x")
                clip_fetch.extract_frames(clip, 10000, [12000])
            args = run.call_args_list[0][0][0]
            self.assertEqual(args[args.index("-ss") + 1], "6.000")

    def test_prune_leaves_clips_a_queued_summary_may_need(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        for name, age in (("old", 5000), ("recent", 100), ("newest", 1)):
            path = os.path.join(d, f"{name}.mp4")
            open(path, "wb").close()
            os.utime(path, (time.time() - age, time.time() - age))
        clip_fetch.prune(d, keep=1, min_age_s=1200)
        self.assertEqual(set(os.listdir(d)), {"recent.mp4", "newest.mp4"})

    def test_prune_keeps_recent_image_dirs_and_cleans_deep_mode_crops(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        for name, age in (("old-images", 5000), ("new-images", 1)):
            path = os.path.join(d, name)
            os.makedirs(path)
            os.utime(path, (time.time() - age, time.time() - age))
        for rel, age in (("describe/old.jpg", 5000), ("describe/new.jpg", 1),
                         ("embed/new.jpg", 1)):
            path = os.path.join(d, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").close()
            os.utime(path, (time.time() - age, time.time() - age))
        clip_fetch.prune(d, keep=1, min_age_s=1200)
        self.assertFalse(os.path.exists(os.path.join(d, "old-images")))
        self.assertTrue(os.path.exists(os.path.join(d, "new-images")))
        self.assertFalse(os.path.exists(os.path.join(d, "describe", "old.jpg")))
        self.assertTrue(os.path.exists(os.path.join(d, "embed", "new.jpg")))

    def _extract_past_the_end(self, last_frame_works):
        import subprocess
        from unittest import mock

        def fake_run(args, **kw):
            if "-sseof" in args and last_frame_works:
                open(args[-1], "wb").close()
            return subprocess.CompletedProcess(args, 0, "", "")

        with tempfile.TemporaryDirectory() as d:
            clip = os.path.join(d, "e.mp4")
            open(clip, "w").close()
            with open(clip + ".start", "w") as f:
                f.write("1000")
            with mock.patch.object(subprocess, "run", side_effect=fake_run) as run:
                frames = clip_fetch.extract_frames(clip, 1000, [26000])
            return frames, run.call_args_list

    def test_key_moment_past_the_end_uses_the_last_frame(self):
        frames, calls = self._extract_past_the_end(True)
        self.assertEqual(list(frames), [26000])
        self.assertIn("-sseof", calls[1][0][0])

    def test_no_frame_when_the_last_frame_fails_too(self):
        frames, calls = self._extract_past_the_end(False)
        self.assertEqual(frames, {})
        self.assertEqual(len(calls), 2)

    def test_prune_missing_dir(self):
        clip_fetch.prune('/nonexistent/dir/xyz')


if __name__ == "__main__":
    unittest.main()