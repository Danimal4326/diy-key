# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for clip_fetch.py: URL construction and clip pruning."""
import os
import shutil
import sys
import tempfile
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

    def test_prune_missing_dir(self):
        clip_fetch.prune('/nonexistent/dir/xyz')


if __name__ == "__main__":
    unittest.main()