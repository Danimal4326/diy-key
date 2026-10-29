#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Outbound mTLS WebSocket ("ucp4") client -- the ongoing connection a real AI
Key opens to the console after ``POST /api/adopt`` (see
``stage1_http/adopt_http.py``, which writes ``adopt_state.json`` for this
script to pick up).

The envelope is two concatenated records, each with an 8-byte header:
``type u8, format u8, compressed u8, reserved u8, length u32 BE``. The length
is 32-bit: a 16-bit read silently truncates any payload over 64 KiB (the
controller reads it at offset 4 as a u32). ``format=3`` means the body is raw
bytes rather than JSON.

AI-Key RPC handlers are the live-confirmed ones from ``../../protocol.md`` §3;
unknown actions are acked with ``{}`` and logged so the console's own
``errors.log`` can teach us new shapes.
"""
import argparse
import json
import logging
import os
import re
import ssl
import struct
import sys
import threading
import time
import uuid

import requests
import urllib3
from websockets.sync.client import connect

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core import config  # noqa: E402
from system_stats import collect_system_metrics  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONSOLE_INTERNAL_API_PORT = 7444

log = logging.getLogger("ucp4-client")

# UCP4 record header (8 bytes): type, format, compressed, reserved, u32 length.
RECORD_TYPE_HEAD = 1
RECORD_TYPE_BODY = 2
RECORD_FORMAT_JSON = 1
RECORD_FORMAT_RAW = 3

# --state-dir (or None for the historical stage1_http/ path). Resolved in
# main(); kept as plain paths here so importing this module has no side effects.
STATE_DIR = None
ADOPT_STATE_FILE = os.path.join(config.DEFAULT_STATE_DIR, "adopt_state.json")
CERT_PATH = os.path.join(config.DEFAULT_STATE_DIR, "server.crt")
KEY_PATH = os.path.join(config.DEFAULT_STATE_DIR, "server.key")


def pack_record(record_type: int, payload) -> bytes:
    if isinstance(payload, (bytes, bytearray)):
        body = bytes(payload)
        rec_format = RECORD_FORMAT_RAW
    else:
        body = json.dumps(payload).encode()
        rec_format = RECORD_FORMAT_JSON
    header = struct.pack(">BBBBI", record_type, rec_format, 0, 0, len(body))
    return header + body


def pack_message(head: dict, body) -> bytes:
    return pack_record(RECORD_TYPE_HEAD, head) + pack_record(RECORD_TYPE_BODY, body)


def unpack_all_records(data: bytes):
    """[(record_type, format, obj)] for every complete record in a frame."""
    records = []
    off = 0
    while off < len(data):
        if len(data) - off < 8:
            log.warning("trailing %d bytes, not enough for a record header", len(data) - off)
            break
        rtype, rec_format, _compressed, _reserved, length = struct.unpack(
            ">BBBBI", data[off:off + 8])
        off += 8
        payload = data[off:off + length]
        off += length
        if rec_format == RECORD_FORMAT_RAW:
            obj = {"_raw_bytes": payload}
        else:
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                obj = {"_raw": payload.decode(errors="replace")}
        records.append((rtype, rec_format, obj))
    return records


def wait_for_adopt_state(fallback_host_port, poll_interval=1.0, fallback_after=10.0):
    """Wait for stage1_http to write adopt_state.json from a fresh adopt POST.

    If the device is already adopted server-side (restart with no fresh POST),
    the controller may never re-POST; after the grace period fall back to a
    tokenless reconnect to the configured console host (the controller's
    record has our pinned fingerprint and accepts that).
    """
    log.info("waiting for %s to appear (written by stage1_http on a successful adopt POST)...",
             ADOPT_STATE_FILE)
    waited = 0.0
    while True:
        if os.path.exists(ADOPT_STATE_FILE):
            state = config.read_adopt_state(STATE_DIR)
            if state.get("hosts"):
                return state
        if fallback_host_port and waited >= fallback_after:
            log.info("no adopt POST seen after %.0fs -- falling back to tokenless reconnect "
                     "to %s (device may already be adopted server-side)",
                     waited, fallback_host_port)
            return {"hosts": [fallback_host_port], "token": None}
        time.sleep(poll_interval)
        waited += poll_interval


def build_ssl_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # per spec: device does not validate the console cert
    ctx.load_cert_chain(CERT_PATH, KEY_PATH)  # our mTLS client identity
    return ctx


def handle_action(head, body, device_info):
    action = head.get("action")
    log.info("<<< controller action=%r head=%s body=%s", action, head, body)

    if action == "getInfo":
        return {
            "mac": device_info["mac_nosep"],
            "type": device_info["type"],
            "version": device_info["version"],
            # Read live so ResetToDefaults/readopt can't leave this stale.
            "adopted": config.is_adopted(STATE_DIR),
            "guid": device_info["guid"],
            "deviceId": device_info["device_id"],
            # The console's host, not ours (used to build upload URLs).
            "connectionHost": device_info.get("console_host") or device_info["ip"],
            "poeType": "unknown",
        }
    if action == "networkStatus":
        return {"ip": device_info["ip"], "connected": True}
    if action == "changeUserPassword":
        log.info("changeUserPassword requested (not persisted in this emulator)")
        return {}
    if action == "systemMetrics":
        # CONFIRMED schema: a flat top-level array of {name, value} numbers.
        raw = collect_system_metrics()
        ram_total = raw["ram"]["totalKilobytes"]
        ram_avail = raw["ram"]["availableKilobytes"]
        ram_used_pct = (100.0 * (ram_total - ram_avail) / ram_total) if ram_total else 0.0
        metrics = [
            {"name": "CPU", "value": raw["cpu"]["load"] or 0.0},
            {"name": "GPU", "value": raw["gpu"]["load"] or 0.0},
            {"name": "Memory", "value": round(ram_used_pct, 1)},
            {"name": "NPU", "value": raw["npu"]["load"] or 0.0},
        ]
        log.info("systemMetrics -> %s", metrics)
        return {"systemMetrics": metrics}
    if action == "getTaskQueueInfo":
        # CONFIRMED schema: six numeric counters. `{}` fails validation silently
        # on every ~5 s poll, so all-zero is the correct idle response.
        return {
            "UI_AUDIO_RAM": 0,
            "UI_AUTO_FACE_ENHANCE": 0,
            "UI_AUTO_RAM": 0,
            "UI_AUTO_STT": 0,
            "UI_MANUAL_FACE_ENHANCE": 0,
            "UI_TASK_NUM": 0,
        }
    if action == "recognizeKeyFrames":
        # Confirmed live: the console asks us to deep-analyse a clip, then we
        # POST the result (multipart field `ram`) to the console's resUrl.
        threading.Thread(
            target=_post_recognize_result,
            args=(device_info["console_host"], device_info["mac_nosep"], body.get("resUrl"),
                  body.get("camera"), body.get("event")),
            daemon=True,
        ).start()
    elif action == "RequestAI":
        # targetUri (":7788/v1/models/<model>/inference") means "call your own
        # local gateway": there is no console callback into :7788.
        threading.Thread(
            target=_post_request_ai_result,
            args=(device_info["mac_nosep"], body),
            daemon=True,
        ).start()
    # Unknown/unimplemented action: ack with an empty body so the controller's
    # RPC doesn't hang, and log it so the next real shape can be captured.
    return {}


def _second_verifier_result(params: dict) -> dict:
    """Same placeholder shape as stage4_modelserver/model_server.py's
    _second_verifier_result. Schema confirmed from the real
    models/second_verifier.py; values stay placeholders until real inference."""
    rows = []
    for entry in params.get("thumbnailMeta") or []:
        ts = entry.get("ts")
        for roi in entry.get("roi", []):
            object_type = str(roi.get("objectType") or "").lower()
            rows.append({
                "thumbnailMs": ts,
                "trackerID": roi.get("trackerID", roi.get("trackerId", "")),
                "objectType": object_type,
                "isValidDetection": True,
                "detectedAs": object_type,
                "modelRunStatus": "skipped",
                "iouScore": None,
                "detectionConfidence": 0.0,
                "detectionConfidenceList": [],
            })
    return {
        "cameraId": params.get("camera", ""),
        "eventId": params.get("event", ""),
        "status": "success",
        "scoreThresholdApplied": params.get("score_threshold", 0.5),
        "preProcessMs": 0,
        "inferMs": 0,
        "timeElapsedMs": 0,
        "verificationResults": rows,
    }


def _callback_headers(mac_nosep):
    return {
        "x-ident": mac_nosep.upper(),
        "x-type": "UP-AI-KEY",
        "x-sysid": "0xa5f0",
    }


def _post_request_ai_result(mac_nosep, task_body):
    target_uri = task_body.get("targetUri", "")
    res_url = task_body.get("resUrl")
    payload = task_body.get("payload", {})
    action = payload.get("action", "classify")
    params = payload.get("params", {})

    model_name = "unknown"
    m = re.search(r"/v1/models/([^/]+)/inference", target_uri)
    if m:
        model_name = m.group(1)

    if model_name == "second_verifier_mlabel":
        result = _second_verifier_result(params)
    else:
        log.warning("RequestAI for unhandled model %r, sending empty result", model_name)
        result = {}

    response_body = {
        "model": model_name,
        "action": action,
        "inference_time_ms": 0.0,
        "result": result,
    }

    if not res_url:
        log.warning("RequestAI task had no resUrl, nothing to post (result=%s)", response_body)
        return
    try:
        resp = requests.post(res_url, headers=_callback_headers(mac_nosep), json=response_body,
                             cert=(CERT_PATH, KEY_PATH), verify=False, timeout=15)
        log.info("POST %s -> %s: %s", res_url, resp.status_code, resp.text[:2000])
    except requests.RequestException as e:
        log.warning("POST %s failed: %r", res_url, e)


def _post_recognize_result(console_host, mac_nosep, res_url, camera_id, event_id):
    """POST our RAM/recognize-anything result to resUrl.

    CONFIRMED from the real firmware's syswrapper.sh: this is multipart/form-data
    with the JSON under form field ``ram`` (optionally with per-tracker JPEG
    fields), not a plain JSON POST. The inner ram shape is still a stub.
    """
    if not res_url:
        log.warning("recognizeKeyFrames task had no resUrl, nothing to post")
        return
    url = res_url if res_url.startswith("http") else \
        f"https://{console_host}:{CONSOLE_INTERNAL_API_PORT}{res_url}"
    stub_ram_result = {
        "cameraId": camera_id,
        "eventId": event_id,
        "status": "success",
        "keyMomentsTags": [],
    }
    files = {"ram": ("ram.json", json.dumps(stub_ram_result), "application/json")}
    try:
        resp = requests.post(url, headers=_callback_headers(mac_nosep), files=files,
                             cert=(CERT_PATH, KEY_PATH), verify=False, timeout=15)
        log.info("POST %s -> %s: %s", url, resp.status_code, resp.text[:2000])
    except requests.RequestException as e:
        log.warning("POST %s failed: %r", url, e)


def _watch_for_reconnect(ws, this_connection_token, stop_event, poll_interval=2.0):
    """A fresh adopt POST means the console wants a new WSS session carrying the
    new token and will not necessarily close the old one. Any token change --
    including clearing on un-adopt -- makes X-Adopted stale, so tear down."""
    while not stop_event.is_set():
        time.sleep(poll_interval)
        current = config.read_token(STATE_DIR)
        if current != this_connection_token:
            log.info("adopt token changed (%s -> %s) -- closing this connection "
                     "to force a fresh one", this_connection_token, current)
            try:
                ws.close()
            except Exception:
                pass
            return


def run(host: str, port: int, token: str, send_token_header: bool, device_info: dict,
        adopted: bool = True):
    device_info = {**device_info, "console_host": host}
    url = f"wss://{host}:{port}/"
    headers = {
        "X-Ident": device_info["mac_nosep"],
        "X-Mode": "0",
        "X-Type": device_info["type"],
        "X-Adopted": "true" if adopted else "false",
        "X-Ip": device_info["ip"],
        "X-Device-Id": device_info["device_id"],
        "X-Version": device_info["version"],
        "X-Guid": device_info["guid"],
    }
    if send_token_header:
        headers["X-Token"] = token

    log.info("connecting to %s with headers=%s", url, headers)
    ctx = build_ssl_context()

    with connect(url, subprotocols=["ucp4"], additional_headers=headers,
                 ssl=ctx, open_timeout=15) as ws:
        log.info("WSS CONNECTED (subprotocol=%s)", ws.subprotocol)

        stop_watch = threading.Event()
        watcher = threading.Thread(target=_watch_for_reconnect, args=(ws, token, stop_watch),
                                   daemon=True)
        watcher.start()

        req_id = str(uuid.uuid4())
        msg = pack_message(
            {"timestamp": int(time.time() * 1000), "type": "request",
             "action": "getConsoleInfo", "id": req_id},
            {},
        )
        ws.send(msg)
        log.info(">>> sent getConsoleInfo request id=%s", req_id)

        try:
            _consume(ws, device_info)
        finally:
            stop_watch.set()


def _consume(ws, device_info):
    for raw in ws:
        if isinstance(raw, str):
            log.warning("unexpected TEXT frame (protocol says binary-only): %r", raw[:200])
            continue
        records = unpack_all_records(raw)
        if len(records) < 2:
            log.warning("frame with <2 records: %s", records)
            continue
        head = records[0][2]
        body = records[1][2]

        if head.get("type") == "response":
            log.info("<<< response to our request id=%s errorCode=%s body=%s",
                     head.get("id"), head.get("errorCode"), body)
            continue

        reply_body = handle_action(head, body, device_info)
        if head.get("id"):
            reply = pack_message(
                {"timestamp": int(time.time() * 1000), "type": "response",
                 "action": head.get("action"), "id": head["id"], "errorCode": 0},
                reply_body,
            )
            ws.send(reply)
            log.info(">>> replied to action=%s id=%s body=%s",
                     head.get("action"), head["id"], reply_body)


def _split_host_port(host_port, default_port=7442):
    if ":" in str(host_port):
        host, port_s = str(host_port).rsplit(":", 1)
        return host, int(port_s)
    return str(host_port), default_port


def main():
    cfg = config.load_config_and_logging(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    config.add_common_flags(ap)
    ap.add_argument("--iface", default=None,
                    help="interface to derive identity from (never bound); defaults to "
                         "cfg [network] iface, else the default-route interface")
    ap.add_argument("--bind-ip", default="0.0.0.0",
                    help="local IP to report as our own (X-Ip etc); outbound client, "
                         "so no socket is bound to it")
    ap.add_argument("--mac", default=None)
    ap.add_argument("--type", default=cfg["platform"],
                    help="catalog model string (ucp4 X-Type)")
    ap.add_argument("--version", default=cfg["fw_version"])
    ap.add_argument("--device-id", default=cfg["device_id"])
    ap.add_argument("--guid", default=cfg["guid"])
    ap.add_argument("--fallback-host-port", default=None,
                    help="host:port to reconnect to (tokenless) if no fresh adopt POST "
                         "shows up within --fallback-after seconds")
    ap.add_argument("--fallback-after", type=float, default=10.0)
    ap.add_argument("--state-dir", default=None,
                    help="where adopt_state.json/server.crt/server.key live -- defaults "
                         "to stage1_http/; set this to isolate another instance")
    args = ap.parse_args()
    args.iface = config.resolve_iface(args.iface, cfg)

    global STATE_DIR, ADOPT_STATE_FILE, CERT_PATH, KEY_PATH
    STATE_DIR = config.resolve_state_dir(args.state_dir)
    paths = config.state_paths(STATE_DIR)
    ADOPT_STATE_FILE, CERT_PATH, KEY_PATH = (
        paths["adopt_state"], paths["cert"], paths["key"])

    device_info = {
        "mac_nosep": config.resolve_mac(args.iface, args.mac, cfg).lower(),
        "type": args.type,
        "version": args.version,
        "device_id": args.device_id,
        "guid": args.guid,
        "ip": config.advertised_ip(config.resolve_ip(args.iface, cfg, args.bind_ip)),
    }

    # Tokens are single-use: send X-Token only the first time a value is seen.
    used_tokens = set()
    backoff = 2

    while True:
        state = wait_for_adopt_state(args.fallback_host_port, fallback_after=args.fallback_after)
        hosts = state.get("hosts") or []
        token = state.get("token")
        # Canonical adopted predicate (token or hosts), matching getInfo,
        # /api/info and discovery.
        adopted = config.is_adopted(STATE_DIR)
        if not hosts:
            log.warning("adopt state had no hosts, retrying in 5s")
            time.sleep(5)
            continue
        try:
            host, port = _split_host_port(hosts[0])
        except (TypeError, ValueError):
            log.warning("malformed adopt state host %r; retrying in 5s", hosts[0])
            time.sleep(5)
            continue

        send_token = token is not None and token not in used_tokens
        if token is not None:
            used_tokens.add(token)
        if token is not None and not send_token:
            log.info("token already used on a prior connection -- reconnecting tokenless "
                     "(fingerprint should already be pinned)")
        try:
            run(host, port, token, send_token, device_info, adopted=adopted)
            log.warning("ucp4 connection closed cleanly by peer; reconnecting in 3s")
            backoff = 2
            time.sleep(3)
        except Exception as e:
            log.warning("connection ended: %r -- reconnecting in %ds", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


if __name__ == "__main__":
    main()
