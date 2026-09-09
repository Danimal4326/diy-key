#!/usr/bin/env python3
"""UBNT L2 discovery-protocol responder (UDP/10001).

Wire format and field IDs are CONFIRMED (not guessed) -- taken from
../PROTOCOL_SPEC.md, a live-verified spec for UniFi Protect Chime
adoption from a sibling reverse-engineering project against a real
controller, cross-checked against the `unifi-discovery` PyPI package.
This is stated there to be the same discovery protocol across all UBNT
device classes (cameras, chimes, APs, switches), not device-specific.

Protocol shape:
  Header (4 bytes): version u8, command u8, data_len u16 big-endian
  Fields (repeated until data_len bytes consumed):
      type u8, length u16 big-endian, data <length> bytes

Versions/commands:
  v1 request/response: version=1, command=0 (same value both ways)
  v2 request:           version=2, command=8
  v2 response:          version=2, command=6

CRITICAL: a response must mirror the request's version/command pair, or
it is silently ignored by the controller with zero observable trace.
"""
import argparse
import json
import logging
import os
import socket
import struct
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ADOPT_STATE_FILE = os.path.join(os.path.dirname(HERE), "stage1_http", "adopt_state.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("discovery-beacon")

DISCOVERY_PORT = 10001

TLV_HW_ADDR = 0x01       # 6 raw MAC bytes
TLV_IP_INFO = 0x02       # 6 bytes MAC + 4 bytes IPv4, can repeat per-interface
TLV_FW_VERSION = 0x03
TLV_MAC_ADDRESS = 0x05   # 6 raw MAC bytes (yes, distinct from hw_addr, same value)
TLV_UPTIME = 0x0A
TLV_HOSTNAME = 0x0B
TLV_PLATFORM = 0x0C      # SKU string -- THE load-bearing field, see module docstring
TLV_SYSID = 0x10         # u16 little-endian
TLV_MODEL = 0x14
TLV_IS_MANAGED = 0x17    # 1 byte: 0 = managed/adopted, nonzero = unmanaged/adoptable
TLV_DEVICE_ID = 0x20


def get_mac(iface: str) -> bytes:
    mac_str = None
    try:
        with open(f"/sys/class/net/{iface}/address") as f:
            mac_str = f.read().strip()
    except OSError:
        pass
    if not mac_str:
        node = uuid.getnode()
        mac_str = ":".join(f"{(node >> ele) & 0xff:02x}" for ele in range(40, -8, -8))
    return bytes.fromhex(mac_str.replace(":", ""))


def get_ip(iface: str) -> bytes:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return socket.inet_aton(ip)


def tlv(t: int, value: bytes) -> bytes:
    return struct.pack(">BH", t, len(value)) + value


def is_adopted() -> bool:
    """Best-effort: adopt_state.json is written by stage1_http/adopt_http.py
    on a successful POST /api/adopt. Its presence (with a token) is our only
    local signal of "the console has adopted us" -- not perfectly reliable
    across reboots/un-adopts, but far better than the previous hardcoded
    "always adoptable" value, which the real console flagged live:
    "AI Key [...] is still adopted by this console but it's adoptable
    according to discovery packet" -- exactly the bug this fixes."""
    try:
        with open(ADOPT_STATE_FILE) as f:
            state = json.load(f)
        return bool(state.get("token")) or bool(state.get("hosts"))
    except (OSError, json.JSONDecodeError):
        return False


def build_response(version: int, command: int, mac: bytes, ip: bytes, hostname: str,
                    fw_version: str, platform: str, sysid: int, uptime_s: int,
                    device_id: str) -> bytes:
    payload = b""
    payload += tlv(TLV_HW_ADDR, mac)
    payload += tlv(TLV_IP_INFO, mac + ip)
    payload += tlv(TLV_MAC_ADDRESS, mac)
    payload += tlv(TLV_FW_VERSION, fw_version.encode())
    payload += tlv(TLV_UPTIME, struct.pack(">I", uptime_s))
    payload += tlv(TLV_HOSTNAME, hostname.encode())
    payload += tlv(TLV_PLATFORM, platform.encode())
    payload += tlv(TLV_MODEL, platform.encode())
    payload += tlv(TLV_SYSID, struct.pack("<H", sysid))
    payload += tlv(TLV_IS_MANAGED, bytes([0x00 if is_adopted() else 0x01]))
    payload += tlv(TLV_DEVICE_ID, device_id.encode())

    header = struct.pack(">BBH", version, command, len(payload))
    return header + payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--hostname", default="aikey-emu")
    ap.add_argument("--platform", default="UP-AI-KEY",
                     help="SKU string matched case/hyphen-insensitively against the "
                          "controller's device catalog. This is our best guess from "
                          "the real firmware's x-type HTTP header value -- if the "
                          "device still doesn't show up as adoptable, try variants "
                          "like 'AI Key' or 'UniFi AI Key'.")
    ap.add_argument("--sysid", type=lambda s: int(s, 0), default=0xa5f0,
                     help="hex sysid, confirmed from real firmware (x-sysid header)")
    ap.add_argument("--fw-version", default="aikey.rk3588.v2.2.6.0.emu.260901.0000")
    ap.add_argument("--device-id", default="00000000-0000-4000-8000-000000000001")
    args = ap.parse_args()

    mac = get_mac(args.iface)
    start = time.time()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("0.0.0.0", DISCOVERY_PORT))
    log.info("Listening for UBNT discovery probes on UDP/%d (mac=%s platform=%r sysid=0x%04x)",
              DISCOVERY_PORT, mac.hex(":"), args.platform, args.sysid)

    while True:
        try:
            data, addr = sock.recvfrom(2048)
        except OSError as e:
            log.warning("recv error: %s", e)
            continue

        log.info("Discovery probe from %s: %s", addr, data.hex())

        if len(data) < 4:
            log.warning("probe too short, ignoring")
            continue
        req_version, req_command = data[0], data[1]

        if (req_version, req_command) == (1, 0):
            resp_version, resp_command = 1, 0
        elif (req_version, req_command) == (2, 8):
            resp_version, resp_command = 2, 6
        else:
            log.warning("unrecognized request version/command (%d/%d), replying v1 anyway",
                        req_version, req_command)
            resp_version, resp_command = 1, 0

        ip = get_ip(args.iface)
        uptime_s = int(time.time() - start)
        resp = build_response(resp_version, resp_command, mac, ip, args.hostname,
                               args.fw_version, args.platform, args.sysid, uptime_s,
                               args.device_id)
        try:
            sock.sendto(resp, addr)
            log.info("Sent v%d/cmd%d discovery response to %s (%d bytes)",
                      resp_version, resp_command, addr, len(resp))
        except OSError as e:
            log.warning("send error to %s: %s", addr, e)


if __name__ == "__main__":
    main()
