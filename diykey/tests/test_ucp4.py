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
from unittest import mock

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
        self.assertEqual(reply["poeType"], "802.3bt")

    def test_get_info_adopted_is_live(self):
        reply = self._call("getInfo")
        self.assertFalse(reply["adopted"])
        self._adopt()
        self.assertTrue(self._call("getInfo")["adopted"])
        config.clear_adopt_state(self.d)
        self.assertFalse(self._call("getInfo")["adopted"])

    def test_network_status(self):
        self.assertEqual(self._call("networkStatus"),
                         {"ip": "10.0.0.9", "connected": True, "linkSpeedMbps": 1000})

    def test_disk_info(self):
        self.assertEqual(self._call("diskInfo"), {"storageSize": "256"})

    def test_get_info_feature_flags(self):
        reply = self._call("getInfo")
        flags = reply["featureFlags"]
        self.assertEqual(flags["supportPersonReId"], {"enabled": True, "version": "v1"})
        self.assertEqual(flags["supportFaceRecognition"], {"enabled": True, "version": "v1"})
        self.assertEqual(flags["aiMode"], "basic")

    def test_model_mode_reported_in_get_info(self):
        self.addCleanup(setattr, ucp4_client, "_ai_mode", "basic")
        self._call("changeAiInferAgentSettings", {"modelMode": "deep"})
        self.assertEqual(self._call("getInfo")["featureFlags"]["aiMode"], "deep")

    def test_change_password_without_pg_socket(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DIYKEY_PG_SOCKET", None)
            self.assertEqual(self._call("changeUserPassword", {"username": "ui", "passwordOld": "a", "passwordNew": "b"}), {})

    def test_pg_password_connects_as_cluster_owner(self):
        # Under the root daemon psql's default role would be "root", which doesn't exist.
        run = mock.Mock(return_value=mock.Mock(returncode=0, stderr=""))
        with mock.patch.dict(os.environ, {"DIYKEY_PG_SOCKET": "/pg", "DIYKEY_PG_USER": "owner"}), \
                mock.patch.object(ucp4_client.subprocess, "run", run):
            ucp4_client._apply_pg_password("new")
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("-U") + 1], "owner")

    def test_pg_password_default_role_without_owner(self):
        run = mock.Mock(return_value=mock.Mock(returncode=0, stderr=""))
        env = {k: v for k, v in os.environ.items() if k != "DIYKEY_PG_USER"}
        env["DIYKEY_PG_SOCKET"] = "/pg"
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(ucp4_client.subprocess, "run", run):
            ucp4_client._apply_pg_password("new")
        self.assertNotIn("-U", run.call_args.args[0])

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


class TestVersionFromUpdateUri(unittest.TestCase):
    def test_version_from_path(self):
        self.assertEqual(ucp4_client._version_from_update_uri(
            "https://h:7443/internal/update/rk3588/AI-Key/firmware/2.2.8/FCECDA000001"),
            "2.2.8")

    def test_version_from_query(self):
        self.assertEqual(ucp4_client._version_from_update_uri(
            "https://h/x?version=2.2.9&mac=1"),
            "2.2.9")

    def test_empty_uri(self):
        self.assertIsNone(ucp4_client._version_from_update_uri(""))


class TestUpdateFirmware(_StateDirTest):
    DEVICE = {
        "mac_nosep": "fcecdaaabbcc",
        "type": "UP-AI-KEY",
        "version": "2.2.6",
        "guid": "00000000-0000-4000-8000-000000000002",
        "device_id": "00000000-0000-4000-8000-000000000001",
        "ip": "10.0.0.9",
    }

    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, ucp4_client, "_latest_fw", None)

    def _call(self, action, body=None):
        return ucp4_client.handle_action({"action": action}, body or {}, dict(self.DEVICE))

    def test_update_pins_version_without_ws(self):
        self.assertEqual(self._call("updateFirmware",
            {"uri": "https://h/internal/update/p/AI-Key/firmware/2.2.8/MAC"}), {})
        self.assertEqual(self._call("getInfo")["version"], "2.2.8")

    def test_update_uses_latest_from_firmware_state(self):
        self._call("firmwareState", {"latestFirmwareVersion": "2.3.0"})
        self.assertEqual(self._call("updateFirmware",
            {"uri": "https://h/blob.bin"}), {})
        self.assertEqual(self._call("getInfo")["version"], "2.3.0")


class TestAdoptStateWait(_StateDirTest):
    def test_returns_existing_state(self):
        self._adopt()
        state = ucp4_client.wait_for_adopt_state(None, poll_interval=0.01, fallback_after=0.01)
        self.assertEqual(state["token"], "tok")

    def test_falls_back_when_no_state(self):
        state = ucp4_client.wait_for_adopt_state("10.0.0.5:7442", poll_interval=0.01,
                                                fallback_after=0.02)
        self.assertEqual(state, {"hosts": ["10.0.0.5:7442"], "token": None})


class TestPgPasswordPlan(unittest.TestCase):
    """changeUserPassword: the rotated password must reach psql, never argv."""

    def _plan(self, env, platform, euid=1000, which=None):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("DIYKEY_PG")}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(ucp4_client.sys, "platform", platform), \
                mock.patch.object(ucp4_client.os, "geteuid", return_value=euid), \
                mock.patch.object(ucp4_client.shutil, "which", side_effect=which):
            return ucp4_client._pg_password_plan("s3cret")

    def test_macos_instance_socket(self):
        cmd, sql = self._plan({"DIYKEY_PG_SOCKET": "/tmp/pg"}, "darwin", which=lambda n: None)
        self.assertEqual(cmd[0], "/opt/homebrew/opt/postgresql@17/bin/psql")
        self.assertEqual(cmd[cmd.index("-h") + 1], "/tmp/pg")
        self.assertIn("ALTER ROLE \"unifi-protect\" PASSWORD 's3cret'", sql)
        self.assertNotIn("s3cret", " ".join(cmd))  # ps would show it

    def test_linux_root_uses_the_system_socket_as_postgres(self):
        cmd, sql = self._plan({}, "linux", euid=0, which=lambda n: {
            "psql": "/usr/bin/psql", "runuser": "/usr/sbin/runuser"}.get(n))
        self.assertEqual(cmd[:4], ["/usr/sbin/runuser", "-u", "postgres", "--"])
        self.assertEqual(cmd[4], "/usr/bin/psql")
        self.assertEqual(cmd[cmd.index("-h") + 1], "/var/run/postgresql")
        self.assertIn("PASSWORD 's3cret'", sql)

    def test_no_postgres_configured(self):
        cmd, reason = self._plan({}, "darwin")
        self.assertIsNone(cmd)
        self.assertIn("DIYKEY_PG_SOCKET", reason)

    def test_embedded_quote_is_doubled(self):
        with mock.patch.dict(os.environ, {"DIYKEY_PG_SOCKET": "/tmp/pg"}, clear=False), \
                mock.patch.object(ucp4_client.sys, "platform", "darwin"):
            _, sql = ucp4_client._pg_password_plan("a'b")
        self.assertIn("PASSWORD 'a''b'", sql)


class TestTaggerConfig(unittest.TestCase):
    def test_missing_model_is_reported_once(self):
        clean = {k: v for k, v in os.environ.items() if k != "DIYKEY_RAM_MODEL"}
        with mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(ucp4_client, "_tagger_warned", False):
            with self.assertLogs("ucp4-client", level="WARNING") as logs:
                self.assertIsNone(ucp4_client._get_tagger())
                self.assertIsNone(ucp4_client._get_tagger())
        self.assertEqual(len([line for line in logs.output if "RAM tagging disabled" in line]), 1)


class TestSplitHostPort(unittest.TestCase):
    def test_with_port(self):
        self.assertEqual(ucp4_client._split_host_port("10.0.0.5:7442"), ("10.0.0.5", 7442))

    def test_default_port(self):
        self.assertEqual(ucp4_client._split_host_port("10.0.0.5"), ("10.0.0.5", 7442))


if __name__ == "__main__":
    unittest.main()
