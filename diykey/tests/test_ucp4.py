# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Unit tests for ucp4_client.py: wire framing, state and RPC handlers."""
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage3_wss")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ucp4_client  # noqa: E402
from core import config  # noqa: E402


class TestPackRecord(unittest.TestCase):
    def test_header_shape(self):
        rec = ucp4_client.pack_record(3, {"a": 1})
        rtype, rec_format, compressed, reserved, length = struct.unpack(">BBBBI", rec[:8])
        self.assertEqual((rtype, rec_format, compressed, reserved), (3, 1, 0, 0))
        self.assertEqual(length, len(rec) - 8)
        self.assertEqual(json.loads(rec[8:]), {"a": 1})

    def test_u32_length_survives_large_payload(self):
        payload = {"blob": "x" * 70000}
        rec = ucp4_client.pack_record(2, payload)
        self.assertEqual(struct.unpack(">I", rec[4:8])[0], len(rec) - 8)
        self.assertEqual(ucp4_client.unpack_all_records(rec)[0][2], payload)

    def test_raw_body_format(self):
        blob = bytes(range(256)) * 400
        rec = ucp4_client.pack_record(2, blob)
        rtype, rec_format, _c, _r, length = struct.unpack(">BBBBI", rec[:8])
        self.assertEqual((rtype, rec_format, length), (2, 3, len(blob)))
        self.assertEqual(rec[8:], blob)
        self.assertEqual(ucp4_client.unpack_all_records(rec)[0][2], {"_raw_bytes": blob})

    def test_pack_message_head_then_body(self):
        data = ucp4_client.pack_message({"h": 1}, {"b": 2})
        recs = ucp4_client.unpack_all_records(data)
        self.assertEqual([r[0] for r in recs], [1, 2])
        self.assertEqual([r[2] for r in recs], [{"h": 1}, {"b": 2}])


class TestUnpack(unittest.TestCase):
    def test_multiple_and_trailing(self):
        data = (ucp4_client.pack_record(1, {"x": "y"})
                + ucp4_client.pack_record(2, {"n": 42})
                + b"\x01\x02\x03")
        recs = ucp4_client.unpack_all_records(data)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[1][2], {"n": 42})

    def test_non_json_payload_is_raw(self):
        body = b"not json"
        data = struct.pack(">BBBBI", 7, 1, 0, 0, len(body)) + body
        self.assertEqual(ucp4_client.unpack_all_records(data), [(7, 1, {"_raw": "not json"})])

    def test_empty(self):
        self.assertEqual(ucp4_client.unpack_all_records(b""), [])


class _StateDirTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self._orig_state = ucp4_client.STATE_DIR
        self._orig_file = ucp4_client.ADOPT_STATE_FILE
        ucp4_client.STATE_DIR = self.d
        ucp4_client.ADOPT_STATE_FILE = os.path.join(self.d, "adopt_state.json")
        self.addCleanup(self._restore)

    def _restore(self):
        ucp4_client.STATE_DIR = self._orig_state
        ucp4_client.ADOPT_STATE_FILE = self._orig_file

    def _adopt(self, token="tok", extra=None):
        config.write_adopt_state(self.d, {"hosts": ["10.0.0.5:7442"], "token": token,
                                          **(extra or {})})


class TestHandleAction(_StateDirTest):
    DEVICE = {
        "mac_nosep": "fcecdaaabbcc",
        "type": "UP-AI-KEY",
        "version": "2.2.6",
        "guid": "00000000-0000-4000-8000-000000000002",
        "device_id": "00000000-0000-4000-8000-000000000001",
        "ip": "10.0.0.9",
    }

    def _call(self, action, body=None):
        return ucp4_client.handle_action({"action": action}, body or {}, dict(self.DEVICE))

    def test_get_info_shape(self):
        self._adopt()
        reply = ucp4_client.handle_action(
            {"action": "getInfo"}, {}, {**self.DEVICE, "console_host": "10.0.0.5"})
        self.assertEqual(reply["mac"], "fcecdaaabbcc")
        self.assertEqual(reply["type"], "UP-AI-KEY")
        self.assertEqual(reply["version"], "2.2.6")
        self.assertTrue(reply["adopted"])
        self.assertEqual(reply["deviceId"], "00000000-0000-4000-8000-000000000001")
        self.assertEqual(reply["connectionHost"], "10.0.0.5")
        self.assertEqual(reply["poeType"], "unknown")

    def test_get_info_adopted_is_live(self):
        reply = self._call("getInfo")
        self.assertFalse(reply["adopted"])
        self._adopt()
        self.assertTrue(self._call("getInfo")["adopted"])
        config.clear_adopt_state(self.d)
        self.assertFalse(self._call("getInfo")["adopted"])

    def test_network_status(self):
        self.assertEqual(self._call("networkStatus"), {"ip": "10.0.0.9", "connected": True})

    def test_task_queue_shape(self):
        reply = self._call("getTaskQueueInfo")
        self.assertEqual(set(reply), {
            "UI_AUDIO_RAM", "UI_AUTO_FACE_ENHANCE", "UI_AUTO_RAM", "UI_AUTO_STT",
            "UI_MANUAL_FACE_ENHANCE", "UI_TASK_NUM"})
        self.assertTrue(all(v == 0 for v in reply.values()))

    def test_system_metrics_flat_array(self):
        reply = self._call("systemMetrics")
        self.assertEqual(len(reply["systemMetrics"]), 4)
        for item in reply["systemMetrics"]:
            self.assertEqual(set(item), {"name", "value"})
            self.assertIsInstance(item["name"], str)
            self.assertIsInstance(item["value"], (int, float))

    def test_unknown_action_is_empty_ack(self):
        self.assertEqual(self._call("sshService"), {})


class TestAdoptStateWait(_StateDirTest):
    def test_returns_existing_state(self):
        self._adopt()
        state = ucp4_client.wait_for_adopt_state(None, poll_interval=0.01, fallback_after=0.01)
        self.assertEqual(state["token"], "tok")

    def test_falls_back_when_no_state(self):
        state = ucp4_client.wait_for_adopt_state("10.0.0.5:7442", poll_interval=0.01,
                                                fallback_after=0.02)
        self.assertEqual(state, {"hosts": ["10.0.0.5:7442"], "token": None})


class TestSplitHostPort(unittest.TestCase):
    def test_with_port(self):
        self.assertEqual(ucp4_client._split_host_port("10.0.0.5:7442"), ("10.0.0.5", 7442))

    def test_default_port(self):
        self.assertEqual(ucp4_client._split_host_port("10.0.0.5"), ("10.0.0.5", 7442))


if __name__ == "__main__":
    unittest.main()
