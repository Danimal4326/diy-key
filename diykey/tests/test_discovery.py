# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Unit tests for the discovery response builder (no sockets)."""
import os
import socket
import struct
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage1_discovery")):
    if p not in sys.path:
        sys.path.insert(0, p)

import beacon  # noqa: E402


def parse_tlvs(payload):
    out = {}
    off = 0
    while off < len(payload):
        t = payload[off]
        length = struct.unpack(">H", payload[off + 1:off + 3])[0]
        out[t] = payload[off + 3:off + 3 + length]
        off += 3 + length
    return out


def build(adopted=False, console_id=None):
    mac = bytes.fromhex("fcecdaaabbcc")
    ip = socket.inet_aton("192.0.2.20")
    with mock.patch.object(beacon.config, "is_adopted", return_value=adopted), \
         mock.patch.object(beacon.config, "console_id_bytes", return_value=console_id):
        return beacon.build_response(
            1, 0, mac, ip, "diy-key", "2.2.6-test",
            "UP-AI-KEY", 0xA5F0, 1234, "device-0001")


class TestTlv(unittest.TestCase):
    def test_tlv_encoding(self):
        self.assertEqual(beacon.tlv(0x0C, b"abc"), b"\x0c\x00\x03abc")
        self.assertEqual(beacon.tlv(0x10, struct.pack("<H", 0xA5F0)),
                         b"\x10\x00\x02\xf0\xa5")
        self.assertEqual(beacon.tlv(0x05, b""), b"\x05\x00\x00")


class TestBuildResponse(unittest.TestCase):
    def test_header_and_fields(self):
        pkt = build()
        version, command, data_len = struct.unpack(">BBH", pkt[:4])
        self.assertEqual((version, command), (1, 0))
        self.assertEqual(data_len, len(pkt) - 4)

        t = parse_tlvs(pkt[4:])
        mac = bytes.fromhex("fcecdaaabbcc")
        self.assertEqual(t[beacon.TLV_HW_ADDR], mac)
        self.assertEqual(t[beacon.TLV_IP_INFO], mac + socket.inet_aton("192.0.2.20"))
        self.assertEqual(struct.unpack(">I", t[beacon.TLV_UPTIME])[0], 1234)
        self.assertEqual(t[beacon.TLV_HOSTNAME], b"diy-key")
        self.assertEqual(t[beacon.TLV_PLATFORM], b"UP-AI-KEY")
        self.assertEqual(t[beacon.TLV_FW_VERSION], b"2.2.6-test")
        self.assertEqual(struct.unpack("<H", t[beacon.TLV_SYSID])[0], 0xA5F0)
        self.assertEqual(t[beacon.TLV_DEVICE_ID], b"device-0001")
        self.assertEqual(t[beacon.TLV_SUPPORT_UCP4], b"\x01")
        self.assertEqual(t[beacon.TLV_DEFAULT_CREDENTIALS], b"\x03")
        # real devices never send these duplicates
        self.assertNotIn(0x05, t)
        self.assertNotIn(0x14, t)

    def test_field_order(self):
        payload = build()[4:]
        order = []
        off = 0
        while off < len(payload):
            order.append(payload[off])
            length = struct.unpack(">H", payload[off + 1:off + 3])[0]
            off += 3 + length
        self.assertEqual(order, [
            beacon.TLV_IP_INFO, beacon.TLV_HW_ADDR, beacon.TLV_UPTIME,
            beacon.TLV_HOSTNAME, beacon.TLV_PLATFORM, beacon.TLV_IS_MANAGED,
            beacon.TLV_FW_VERSION, beacon.TLV_SYSID, beacon.TLV_DEVICE_ID,
            beacon.TLV_SUPPORT_UCP4, beacon.TLV_DEFAULT_CREDENTIALS,
        ])

    def test_is_managed_is_four_bytes(self):
        for adopted, expected in ((False, 1), (True, 0)):
            raw = parse_tlvs(build(adopted=adopted)[4:])[beacon.TLV_IS_MANAGED]
            self.assertEqual(len(raw), 4)
            self.assertEqual(struct.unpack(">I", raw)[0], expected)

    def test_controller_id_only_when_adopted(self):
        self.assertNotIn(beacon.TLV_CONTROLLER_ID, parse_tlvs(build()[4:]))
        cid = bytes(range(16))
        self.assertEqual(parse_tlvs(build(adopted=True, console_id=cid)[4:])
                         [beacon.TLV_CONTROLLER_ID], cid)


class TestResponseVersion(unittest.TestCase):
    def test_v1_roundtrip(self):
        self.assertEqual(beacon.response_version(1, 0), (1, 0))

    def test_v2_roundtrip(self):
        self.assertEqual(beacon.response_version(2, 8), (2, 6))

    def test_unknown_falls_back_to_v1(self):
        with mock.patch.object(beacon.log, "warning"):
            self.assertEqual(beacon.response_version(2, 9), (1, 0))


if __name__ == "__main__":
    unittest.main()
