#!/usr/bin/env python3
"""Outbound mTLS WebSocket ("ucp4") client -- the ongoing connection a real
AI Key opens to the console immediately after a successful HTTP /api/adopt
(see ../stage1_http/adopt_http.py, which is what receives the adopt POST
and writes ../stage1_http/adopt_state.json for this script to pick up).

Wire protocol CONFIRMED (live-verified against a real controller, for a
UniFi Protect Chime) in ../PROTOCOL_SPEC.md section 3-4. AI Key's HTTP
adopt handshake (stage1_http) already matched that spec byte-for-byte
against our real UNVR, so this reuses the same ucp4 envelope/header
contract; the RPC action set (section 5 of the spec) is chime-specific and
NOT assumed correct here -- this client logs every inbound action verbatim
so the real controller teaches us the aiprocessor-specific ones live.
"""
import argparse
import json
import logging
import os
import re
import ssl
import struct
import subprocess
import threading
import time
import uuid

import requests
import urllib3
import websockets
from websockets.sync.client import connect

from system_stats import collect_system_metrics

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONSOLE_INTERNAL_API_PORT = 7444
MODEL_SERVER_PORT = 7788

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ucp4-client")

HERE = os.path.dirname(os.path.abspath(__file__))
ADOPT_STATE_FILE = os.path.join(os.path.dirname(HERE), "stage1_http", "adopt_state.json")
CERT_PATH = os.path.join(os.path.dirname(HERE), "stage1_http", "server.crt")
KEY_PATH = os.path.join(os.path.dirname(HERE), "stage1_http", "server.key")


def get_mac(iface: str) -> str:
    try:
        with open(f"/sys/class/net/{iface}/address") as f:
            return f.read().strip().lower()
    except OSError:
        node = uuid.getnode()
        return ":".join(f"{(node >> ele) & 0xff:02x}" for ele in range(40, -8, -8))


def get_ip(iface: str) -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        s.close()


def pack_record(record_type: int, payload: dict) -> bytes:
    body = json.dumps(payload).encode()
    header = struct.pack(">BB4sH", record_type, 1, b"\x00\x00\x00\x00", len(body))
    return header + body


def pack_message(head: dict, body: dict) -> bytes:
    return pack_record(1, head) + pack_record(2, body)


def unpack_all_records(data: bytes):
    records = []
    off = 0
    while off < len(data):
        if len(data) - off < 8:
            log.warning("trailing %d bytes, not enough for a record header", len(data) - off)
            break
        rtype, version, reserved, length = struct.unpack(">BB4sH", data[off:off + 8])
        off += 8
        payload = data[off:off + length]
        off += length
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            obj = {"_raw": payload.decode(errors="replace")}
        records.append((rtype, version, obj))
    return records


def wait_for_adopt_state(fallback_host_port, poll_interval=1.0, fallback_after=10.0):
    """Waits for stage1_http to write adopt_state.json from a fresh POST
    /api/adopt (the normal, token-bearing first-adoption path). But if the
    device already knows it's adopted server-side (e.g. we restarted and
    lost our in-memory state, but the controller's own record of us is
    still isAdopted=true), the controller may never re-POST /api/adopt --
    per PROTOCOL_SPEC.md section 3, a reconnect after adoption doesn't
    need a token at all ("device record is marked adopted but has no
    pinned fingerprint yet: pin this one and accept"). So after a short
    grace period with no fresh adopt POST, fall back to reconnecting
    tokenless to the last-known console host.
    """
    log.info("waiting for %s to appear (written by stage1_http on a successful adopt POST)...",
              ADOPT_STATE_FILE)
    waited = 0.0
    while True:
        if os.path.exists(ADOPT_STATE_FILE):
            try:
                with open(ADOPT_STATE_FILE) as f:
                    state = json.load(f)
                if state.get("hosts"):
                    return state
            except (OSError, json.JSONDecodeError):
                pass
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
    ctx.verify_mode = ssl.CERT_NONE  # per spec: device does not validate controller's cert
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
            "adopted": True,
            "guid": device_info["guid"],
            "deviceId": device_info["device_id"],
            "connectionHost": device_info["ip"],
            # CONFIRMED required (z.string()) via a live Zod validation error
            # in the real controller's errors.log ("poeType": "Required") --
            # "unknown" is a value already seen for real in this device's own
            # bootstrap record (featureFlags.poeType), so it's accepted.
            "poeType": "unknown",
        }
    if action == "networkStatus":
        return {"ip": device_info["ip"], "connected": True}
    if action == "changeUserPassword":
        log.info("changeUserPassword requested (not persisted in this emulator)")
        return {}
    if action == "systemMetrics":
        # CONFIRMED exact schema via the real controller's own source
        # (service.js): fetchMetricsSchema = z.object({
        #   systemMetrics: z.object({name: z.union([nativeEnum, z.string()]),
        #                            value: z.number()}).array(),
        #   aiFeatureMetrics: z.record(z.string(), z.number()).optional()
        # }). Our first attempt (nested {cpu:{load,temperature},...}) failed
        # validation server-side with "systemMetrics: Required" (it wants a
        # top-level array under that exact key) -- found via the real
        # controller's own errors.log after a live capture.
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
        # CONFIRMED exact schema via service.js: taskQueueInfoSchema =
        # z.object({UI_AUDIO_RAM, UI_AUTO_FACE_ENHANCE, UI_AUTO_RAM,
        # UI_AUTO_STT, UI_MANUAL_FACE_ENHANCE, UI_TASK_NUM} -- all
        # z.number()). We had been replying `{}`, which silently failed
        # this same validation on every single 5s poll the whole time.
        # All zero (no real task queue yet) is a valid response.
        return {
            "UI_AUDIO_RAM": 0,
            "UI_AUTO_FACE_ENHANCE": 0,
            "UI_AUTO_RAM": 0,
            "UI_AUTO_STT": 0,
            "UI_MANUAL_FACE_ENHANCE": 0,
            "UI_TASK_NUM": 0,
        }
    if action == "recognizeKeyFrames":
        # CONFIRMED live (real vehicle-detection event on a real camera):
        # this is the actual "please do deep analysis on this clip" task.
        # reqUrl/resUrl are relative paths under the console's own internal
        # API (https://<console-ip>:7444). We fetch the referenced clip
        # from reqUrl and are expected to POST our analysis result to
        # resUrl. Response *schema* for resUrl is NOT yet confirmed -- this
        # posts a placeholder body purely to capture the resulting Zod
        # validation error from the console's errors.log, the same
        # technique that solved systemMetrics/getTaskQueueInfo.
        threading.Thread(
            target=_post_recognize_result,
            args=(device_info["console_host"], device_info["mac_nosep"], body.get("resUrl"),
                  body.get("camera"), body.get("event")),
            daemon=True,
        ).start()
    elif action == "RequestAI":
        # CONFIRMED live, but our first reading of targetUri was backwards:
        # a real event (2026-09-03 22:53:49Z) dispatched this and the
        # console's own service.log shows NO inbound connection ever
        # reaching our :7788 (stage4_modelserver never logged a hit) --
        # so the console does not call us. targetUri (":7788/v1/models/
        # <model>/inference", no host) means "call your OWN localhost
        # service", mirroring ui-infer-gateway's own main.py: it's a local
        # HTTP client that's meant to hit itself and let the gateway code
        # own the resUrl callback. Since we don't have a real gateway
        # running real models yet, do that step ourselves here: compute
        # the same stub result stage4_modelserver/model_server.py would
        # return, and POST it (plain JSON, per callback.py -- NOT the
        # multipart form recognize-anything/RAM needs) to resUrl.
        threading.Thread(
            target=_post_request_ai_result,
            args=(device_info["mac_nosep"], body),
            daemon=True,
        ).start()
    # Unknown/unimplemented action: ack with empty body so we don't hang
    # the controller's RPC, and we get to see what it does next.
    return {}


def _second_verifier_result(params: dict) -> dict:
    """Same placeholder shape as stage4_modelserver/model_server.py's
    _second_verifier_result -- kept in sync manually (small enough that a
    shared-module import isn't worth the cross-directory wiring). CONFIRMED
    exact schema from the real models/second_verifier.py docstring; values
    are placeholders (modelRunStatus "skipped", detector's claimed class
    echoed back) until real inference is wired up."""
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
    headers = {
        "x-ident": mac_nosep.upper(),
        "x-type": "UP-AI-KEY",
        "x-sysid": "0xa5f0",
    }
    try:
        resp = requests.post(res_url, headers=headers, json=response_body,
                              cert=(CERT_PATH, KEY_PATH), verify=False, timeout=15)
        log.info("POST %s -> %s: %s", res_url, resp.status_code, resp.text[:2000])
    except requests.RequestException as e:
        log.warning("POST %s failed: %r", res_url, e)


def _post_recognize_result(console_host, mac_nosep, res_url, camera_id, event_id):
    """POST our RAM/recognize-anything result to resUrl.

    CONFIRMED from the real firmware's syswrapper.sh (upload_ram()): this is
    NOT a plain JSON POST (unlike the reverification path) -- it's
    multipart/form-data with the JSON result under form field name `ram`
    (`-F "ram=@<file>;type=application/json"`), optionally with additional
    per-tracker snapshot IMAGE fields (field name = trackerID) for
    face/lpr/pose results. Headers confirmed identical to every other
    console-callback in this codebase: x-ident (MAC, no separators,
    uppercase), x-type: UP-AI-KEY, x-sysid: 0xa5f0, mTLS client cert,
    verify=False (self-signed console cert). The console's own response body
    is itself JSON shaped {"ram": {"status": ...}} (syswrapper reads
    `.ram.status` from it).

    The exact `ram` JSON shape is NOT yet confirmed -- this sends a
    minimal-but-structurally-plausible stub (informed by the ramDetections
    Postgres schema: eventId/cameraId identifiers, tag/label arrays) purely
    to capture the real validation error, the same technique that solved
    systemMetrics/getTaskQueueInfo/getInfo.
    """
    if not res_url:
        log.warning("recognizeKeyFrames task had no resUrl, nothing to post")
        return
    url = res_url if res_url.startswith("http") else f"https://{console_host}:{CONSOLE_INTERNAL_API_PORT}{res_url}"
    headers = {
        "x-ident": mac_nosep.upper(),
        "x-type": "UP-AI-KEY",
        "x-sysid": "0xa5f0",
    }
    stub_ram_result = {
        "cameraId": camera_id,
        "eventId": event_id,
        "status": "success",
        "keyMomentsTags": [],
    }
    files = {"ram": ("ram.json", json.dumps(stub_ram_result), "application/json")}
    try:
        resp = requests.post(url, headers=headers, files=files, cert=(CERT_PATH, KEY_PATH),
                              verify=False, timeout=15)
        log.info("POST %s -> %s: %s", url, resp.status_code, resp.text[:2000])
    except requests.RequestException as e:
        log.warning("POST %s failed: %r", url, e)


def _read_current_token():
    if not os.path.exists(ADOPT_STATE_FILE):
        return None
    try:
        with open(ADOPT_STATE_FILE) as f:
            return json.load(f).get("token")
    except (OSError, json.JSONDecodeError):
        return None


def _watch_for_reconnect(ws, this_connection_token, stop_event, poll_interval=2.0):
    """A fresh POST /api/adopt (e.g. after the console-side device was
    removed and re-adopted) means the controller wants a brand-new WSS
    session carrying the new token -- it will NOT necessarily close our
    existing connection first. Left alone, this process would sit forever
    on the old session, the controller would wait forever for a session
    bearing the new token, and the device would show "instantly offline"
    right after a fresh adopt. So: watch adopt_state.json for a token
    different from the one this connection was opened with, and force a
    reconnect the moment one shows up.
    """
    while not stop_event.is_set():
        time.sleep(poll_interval)
        current = _read_current_token()
        if current and current != this_connection_token:
            log.info("new adopt token detected (%s -> %s) -- closing this connection "
                      "to force a fresh one", this_connection_token, current)
            try:
                ws.close()
            except Exception:
                pass
            return


def run(host: str, port: int, token: str, send_token_header: bool, device_info: dict):
    device_info = {**device_info, "console_host": host}
    url = f"wss://{host}:{port}/"
    headers = {
        "X-Ident": device_info["mac_nosep"],
        "X-Mode": "0",
        "X-Type": device_info["type"],
        "X-Adopted": "true",
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

        # send our own getConsoleInfo request right away, per spec
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
        (htype, hver, head), (btype, bver, body) = records[0], records[1]

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
                      head.get("action"), head.get("id"), reply_body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--type", default="UP-AI-KEY")
    ap.add_argument("--version", default="2.2.6")
    ap.add_argument("--device-id", default="00000000-0000-4000-8000-000000000001")
    ap.add_argument("--guid", default="00000000-0000-4000-8000-000000000002")
    ap.add_argument("--fallback-host-port", default=None,
                     help="e.g. 192.0.2.5:7442 -- used for a tokenless reconnect if no fresh "
                          "adopt POST shows up within --fallback-after seconds (see "
                          "wait_for_adopt_state's docstring)")
    ap.add_argument("--fallback-after", type=float, default=10.0)
    args = ap.parse_args()

    mac_nosep = get_mac(args.iface).replace(":", "")
    device_info = {
        "mac_nosep": mac_nosep,
        "type": args.type,
        "version": args.version,
        "device_id": args.device_id,
        "guid": args.guid,
        "ip": get_ip(args.iface),
    }

    last_used_token = None
    backoff = 2
    while True:
        state = wait_for_adopt_state(args.fallback_host_port, fallback_after=args.fallback_after)
        host_port = state["hosts"][0]
        host, port_s = host_port.rsplit(":", 1)
        port = int(port_s)
        # This is the real, current token per adopt_state.json (None if no adopt
        # POST has ever landed and we're on the tokenless fallback path) -- it's
        # the baseline the watcher thread compares future file changes against,
        # so it must NOT be artificially suppressed to None on reconnects (that
        # bug caused an infinite reconnect loop: the watcher would see the file's
        # real, unchanged token as "new" relative to a suppressed None baseline,
        # on every single reconnect).
        current_token = state.get("token")
        send_token_header = current_token is not None and current_token != last_used_token
        try:
            run(host, port, current_token, send_token_header, device_info)
            last_used_token = current_token
            backoff = 2
        except Exception as e:
            log.warning("WSS connection ended/failed: %r -- reconnecting in %ds", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


if __name__ == "__main__":
    main()
