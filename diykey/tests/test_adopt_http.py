# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Unit tests for adopt_http.py's state handling and cert generation."""
import json
import os
import shutil
import ssl
import stat
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage1_http")):
    if p not in sys.path:
        sys.path.insert(0, p)

import adopt_http  # noqa: E402
from core import config  # noqa: E402


class TestPersistAdopt(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def _state(self):
        with open(config.adopt_state_path(self.d)) as f:
            return json.load(f)

    def test_valid_payload_persists(self):
        payload = {"hosts": ["10.0.0.5:7442"], "token": "tok", "protocol": "wss"}
        self.assertTrue(adopt_http.persist_adopt(self.d, payload))
        self.assertEqual(self._state(), payload)
        self.assertEqual(stat.S_IMODE(os.stat(config.adopt_state_path(self.d)).st_mode), 0o600)

    def test_token_only_is_accepted(self):
        self.assertTrue(adopt_http.persist_adopt(self.d, {"token": "tok"}))
        self.assertTrue(config.is_adopted(self.d))

    def test_empty_payload_rejected(self):
        self.assertFalse(adopt_http.persist_adopt(self.d, {}))
        self.assertFalse(adopt_http.persist_adopt(self.d, {"hosts": [], "token": ""}))
        self.assertFalse(config.is_adopted(self.d))

    def test_rejected_payload_keeps_existing_state(self):
        adopt_http.persist_adopt(self.d, {"hosts": ["10.0.0.5:7442"], "token": "old"})
        self.assertFalse(adopt_http.persist_adopt(self.d, {"username": "ubnt"}))
        self.assertEqual(config.read_token(self.d), "old")


class TestEnsureCert(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_generates_once_and_is_reusable(self):
        cert, key = adopt_http.ensure_cert(self.d)
        self.assertTrue(os.path.exists(cert))
        self.assertTrue(os.path.exists(key))
        self.assertEqual(stat.S_IMODE(os.stat(key).st_mode), 0o600)

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)  # parses the keypair

        # A second call must not regenerate: the console pins the fingerprint.
        before = os.stat(cert).st_mtime_ns
        adopt_http.ensure_cert(self.d)
        self.assertEqual(os.stat(cert).st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()
