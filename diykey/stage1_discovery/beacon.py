#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""UBNT L2 discovery responder (UDP/10001) for the AI Key emulator.

Wire format and the field set/order are CONFIRMED against live G3 captures,
the controller's own ``service.js`` parser and ``pyunifiwire`` (see
``../../protocol.md`` §1): header ``version u8, command u8, data_len u16 BE``,
then ``type u8, length u16 BE, value`` TLVs. A response must mirror the
request's version/command pair or the controller silently ignores it.

Platform TLV 0x0C is the load-bearing field: the controller matches it
against its SKU catalogue (``UP-AI-KEY`` for this device class).
"""
import argparse
import logging
import os
import socket
import struct
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core import config  # noqa: E402

log = logging.getLogger("discovery-beacon")

DISCOVERY_PORT = 10001
# The controller scans this group (not broadcast); joining also programs the
# NIC's multicast filter.
MCAST_GROUP = "233.89.188.1"

TLV_HW_ADDR = 0x01
TLV_IP_INFO = 0x02
TLV_FW_VERSION = 0x03
TLV_UPTIME = 0x0A
TLV_HOSTNAME = 0x0B
TLV_PLATFORM = 0x0C
TLV_SYSID = 0x10
TLV_IS_MANAGED = 0x17
TLV_DEVICE_ID = 0x20
# Only present once adopted: 16 raw bytes of the console's anonymousDeviceId.
TLV_CONTROLLER_ID = 0x26
# Real cameras send a 1-byte DEFAULT_CREDENTIALS (3 observed).
TLV_DEFAULT_CREDENTIALS = 0x2C
# Gates needUpdateBeforeAdoption.
TLV_SUPPORT_UCP4 = 0x3F
# Do NOT send 0x05 (mac_address) or 0x14 (model): real devices never do, and
# 0x2B (guid) is a console-only identity field.

STATE_DIR = None


def tlv(t: int, value: bytes) -> bytes:
    return struct.pack(">BH", t, len(value)) + value


def build_response(version: int, command: int, mac: bytes, ip: bytes, hostname: str,
                   fw_version: str, platform: str, sysid: int, uptime_s: int,
                   device_id: str) -> bytes:
    payload = b""
    payload += tlv(TLV_IP_INFO, mac + ip)
    payload += tlv(TLV_HW_ADDR, mac)
    payload += tlv(TLV_UPTIME, struct.pack(">I", uptime_s))
    payload += tlv(TLV_HOSTNAME, hostname.encode())
    payload += tlv(TLV_PLATFORM, platform.encode())
    payload += tlv(TLV_IS_MANAGED, struct.pack(">I", 0 if config.is_adopted(STATE_DIR) else 1))
    payload += tlv(TLV_FW_VERSION, fw_version.encode())
    payload += tlv(TLV_SYSID, struct.pack("<H", sysid))
    payload += tlv(TLV_DEVICE_ID, device_id.encode())
    console_id = config.console_id_bytes(STATE_DIR)
    if console_id:
        payload += tlv(TLV_CONTROLLER_ID, console_id)
    payload += tlv(TLV_SUPPORT_UCP4, bytes([0x01]))
    payload += tlv(TLV_DEFAULT_CREDENTIALS, bytes([0x03]))

    header = struct.pack(">BBH", version, command, len(payload))
    return header + payload


def response_version(req_version: int, req_command: int):
    """Mirror the probe's version/command pair; unknown probes get a v1 reply."""
    if (req_version, req_command) == (1, 0):
        return 1, 0
    if (req_version, req_command) == (2, 8):
        return 2, 6
    log.warning("unrecognized request version/command (%d/%d), replying v1 anyway",
                req_version, req_command)
    return 1, 0


def main():
    cfg = config.load_config_and_logging(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    config.add_common_flags(ap)
    ap.add_argument("--iface", default=None,
                    help="interface to bind/derive identity from; defaults to cfg "
                         "[network] iface, else the default-route interface (auto-detected)")
    ap.add_argument("--bind-ip", default="0.0.0.0",
                    help="IP to advertise in discovery replies; the socket still binds "
                         "0.0.0.0 so broadcasts arrive (use --iface to scope reception)")
    ap.add_argument("--mac", default=None, help="override MAC (hex, with or without colons)")
    ap.add_argument("--hostname", default=cfg["hostname"])
    ap.add_argument("--platform", default=cfg.get("discovery_platform") or cfg["platform"],
                    help="discovery platform TLV 0x0C; must be the SKU string the "
                         "controller's catalogue knows ('UP-AI-KEY')")
    ap.add_argument("--sysid", type=lambda s: int(s, 0), default=config.sysid_int(cfg),
                    help="hex hardware/model id (0xa5f0 for AI Key)")
    ap.add_argument("--fw-version", default=cfg["discovery_fw_version"])
    ap.add_argument("--device-id", default=cfg["device_id"])
    ap.add_argument("--state-dir", default=None,
                    help="where adopt_state.json lives -- defaults to stage1_http/; set "
                         "this to isolate a second instance (see core/instance_manager.py)")
    args = ap.parse_args()
    args.iface = config.resolve_iface(args.iface, cfg)

    global STATE_DIR
    STATE_DIR = config.resolve_state_dir(args.state_dir)

    mac = bytes.fromhex(config.resolve_mac(args.iface, args.mac, cfg))
    ip = config.advertised_ip(config.resolve_ip(args.iface, cfg, args.bind_ip))
    ip_bytes = socket.inet_aton(ip)
    start = time.time()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # Scope to this interface so one instance can't answer another's probes.
    # SO_BINDTODEVICE is Linux-only; macOS/BSD simply leave the socket unscoped.
    bind_device = getattr(socket, "SO_BINDTODEVICE", None)
    if bind_device is None:
        log.info("SO_BINDTODEVICE not available on this platform; discovery socket "
                 "is not interface-scoped")
    else:
        try:
            sock.setsockopt(socket.SOL_SOCKET, bind_device, args.iface.encode() + b"\0")
        except OSError as e:
            log.warning("SO_BINDTODEVICE(%s) failed: %s -- this instance may see/answer "
                        "discovery probes meant for other instances on the same box",
                        args.iface, e)
    sock.bind(("0.0.0.0", DISCOVERY_PORT))

    # Join scoped to this interface; required to receive multicast and to program
    # the NIC filter on multi-instance hosts.
    try:
        if_index = socket.if_nametoindex(args.iface)
        try:
            # Linux: struct ip_mreqn (group, 0.0.0.0, ifindex).
            mreq = socket.inet_aton(MCAST_GROUP) + socket.inet_aton("0.0.0.0") + \
                struct.pack("@i", if_index)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError:
            # BSD/macOS: struct ip_mreq carries the interface address, not an index.
            mreq = socket.inet_aton(MCAST_GROUP) + socket.inet_aton(ip)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        log.info("Joined multicast group %s on %s (ifindex %d)",
                 MCAST_GROUP, args.iface, if_index)
    except OSError as e:
        log.warning("failed to join multicast group %s on %s: %s -- this instance "
                    "will only be discoverable via unicast re-probes of an already-"
                    "known IP, not the initial adopt-candidate scan",
                    MCAST_GROUP, args.iface, e)

    log.info("Listening for UBNT discovery probes on %s (mac=%s platform=%r sysid=0x%04x)",
             args.iface, mac.hex(":"), args.platform, args.sysid)

    while True:
        try:
            data, addr = sock.recvfrom(2048)
        except OSError as e:
            log.warning("recv error: %s", e)
            continue

        log.debug("Discovery probe from %s: %s", addr, data.hex())

        if len(data) < 4:
            log.warning("probe too short, ignoring")
            continue
        resp_version, resp_command = response_version(data[0], data[1])

        uptime_s = int(time.time() - start)
        # Read live: a simulated firmware update pins a new version in the state dir.
        fw_version = config.effective_discovery_fw_version(
            {"discovery_fw_version": args.fw_version}, STATE_DIR)
        resp = build_response(resp_version, resp_command, mac, ip_bytes, args.hostname,
                              fw_version, args.platform, args.sysid, uptime_s,
                              args.device_id)
        try:
            sock.sendto(resp, addr)
            log.debug("Sent v%d/cmd%d discovery response to %s (%d bytes)",
                      resp_version, resp_command, addr, len(resp))
        except OSError as e:
            log.warning("send error to %s: %s", addr, e)


if __name__ == "__main__":
    main()
