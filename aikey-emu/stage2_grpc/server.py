#!/usr/bin/env python3
"""Stub ProtectAPI gRPC server, byte-compatible with the real AI Key's
uaibd/grpcserver/grpc_server.py (same .proto contract, copied from the
real firmware's ui-uaibd package). Binds plaintext on :50051, exactly like
the real device (add_insecure_port) -- so behaviorally we match it, warts
and all.

Run generate_pb2.sh first to produce stage2_grpc/generated/.
"""
import argparse
import logging
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))

import grpc
from concurrent import futures

from unifi.protect.v1 import api_pb2, api_pb2_grpc, types_pb2 as protect_types_pb2
from unifi.core.console.v1 import types_pb2 as console_types_pb2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aikey-grpc-stub")


def get_mac(iface: str) -> str:
    try:
        with open(f"/sys/class/net/{iface}/address") as f:
            return f.read().strip().upper().replace(":", "")
    except OSError:
        node = uuid.getnode()
        return "".join(f"{(node >> ele) & 0xff:02x}" for ele in range(40, -8, -8)).upper()


class StubAIKeyService(api_pb2_grpc.ProtectAPIServicer):
    """Mirrors uaibd's AIKeyService: constant/faked state, streamed at ~1Hz."""

    def __init__(self, hostname: str, mac: str, fw_version: str):
        self.hostname = hostname
        self.mac = mac
        self.fw_version = fw_version
        self.lock = threading.Lock()
        self.state = console_types_pb2.STATE_RUNNING
        log.info("StubAIKeyService up: hostname=%s mac=%s fw=%s", hostname, mac, fw_version)

    def Cameras(self, request, context):
        log.info("Cameras() called by %s", context.peer())
        yield api_pb2.CamerasResponse(cameras=[])

    def Detections(self, request, context):
        log.info("Detections() stream started by %s", context.peer())
        while context.is_active():
            yield api_pb2.DetectionsResponse(
                classified_count=0,
                queued_count=0,
                classifications=[],
            )
            time.sleep(1)

    def DeviceStatus(self, request, context):
        log.info("DeviceStatus() stream started by %s", context.peer())
        while context.is_active():
            with self.lock:
                response = api_pb2.DeviceStatusResponse(
                    state=self.state,
                    console_info=protect_types_pb2.ConsoleInfo(name="", ip=""),
                    firmware_version=console_types_pb2.FirmwareVersionInfo(
                        release_channel=console_types_pb2.RELEASE_CHANNEL_ALPHA,
                        current=self.fw_version,
                    ),
                    settings=console_types_pb2.Settings(
                        name=self.hostname,
                        timezone="Etc/UTC",
                        lcm_settings=console_types_pb2.LcmSettings(
                            enabled=True, brightness_percent=100, color="0075FF"
                        ),
                        is_locating=False,
                    ),
                )
            yield response
            time.sleep(1)

    def RestartDevice(self, request, context):
        log.warning("RestartDevice() requested by %s -- NOT actually rebooting (stub)",
                    context.peer())
        return api_pb2.RestartDeviceResponse()

    def UpdateDevice(self, request, context):
        log.warning("UpdateDevice() (OTA trigger) requested by %s -- ignored (stub)",
                    context.peer())
        return api_pb2.UpdateDeviceResponse()

    def UpdateSettings(self, request, context):
        log.info("UpdateSettings() from %s: mask=%s settings=%s",
                  context.peer(), list(request.mask.paths), request.settings)
        return api_pb2.UpdateSettingsResponse()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--hostname", default="aikey-emu")
    ap.add_argument("--fw-version", default="aikey.rk3588.v2.2.6.0.emu.260901.0000")
    args = ap.parse_args()

    mac = get_mac(args.iface)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    api_pb2_grpc.add_ProtectAPIServicer_to_server(
        StubAIKeyService(args.hostname, mac, args.fw_version), server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    log.info("gRPC ProtectAPI stub listening on [::]:%d", args.port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
