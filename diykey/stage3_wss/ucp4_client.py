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
import shutil
import ssl
import struct
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import parse_qsl, urlsplit

import requests
import urllib3
from websockets.sync.client import connect

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core import config  # noqa: E402
from system_stats import collect_system_metrics  # noqa: E402
import clip_fetch  # noqa: E402
import firmware  # noqa: E402
import ram_tagger  # noqa: E402
import nl_search  # noqa: E402
import clip_embed  # noqa: E402
import second_verifier  # noqa: E402

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

# What a real AI Key advertises in getInfo featureFlags (Protect's aiprocessor model lists
# them); the UI enables features, e.g. person second-stage verification, from these.
_FEATURE = {"enabled": True, "version": "v1"}
FEATURE_FLAGS = {k: dict(_FEATURE) for k in (
    "supportPoseEstimation", "supportPersonReId", "supportFaceEnhancement",
    "supportRetroactiveProcessing", "supportImageSearch", "supportRecognizeAnything",
    "supportFaceRecognition", "supportLicensePlateRecognition")}
# Set by changeAiInferAgentSettings {modelMode}; Protect reads it back as featureFlags.aiMode.
_ai_mode = "basic"
# latestFirmwareVersion from the console's last firmwareState: the update target when the
# updateFirmware uri doesn't name one.
_latest_fw = None
# How long a simulated firmware update "installs" before the fake reboot.
FAKE_UPDATE_S = float(os.environ.get("DIYKEY_FAKE_UPDATE_S") or 15)


def _fw_version(device_info):
    """Reported firmware: the version a simulated update pinned, else the configured one."""
    return config.pinned_fw_version(STATE_DIR) or device_info["version"]


def _version_from_update_uri(uri):
    """Target version out of the console's update URL, or None: service.js builds
    /internal/update/<platform>/<product>/<firmware|software>/<version>/<mac>, or the
    same fields as a query string (getUpdateUrl)."""
    if not uri:
        return None
    try:
        parts = urlsplit(uri)
    except ValueError:
        return None
    segments = [seg for seg in parts.path.split("/") if seg]
    if len(segments) >= 3 and segments[-3] in ("firmware", "software"):
        return segments[-2]
    return dict(parse_qsl(parts.query)).get("version") or None


def _simulate_update(ws, target, body, mac_nosep):
    """Keep a copy of the firmware (unless DIYKEY_FW_DOWNLOAD=0), pretend to flash, pin the
    new version, then drop the connection as the reboot. The reconnect reports the pinned
    version, which ends Protect's update. Nothing is ever installed."""
    t0 = time.monotonic()
    if os.environ.get("DIYKEY_FW_DOWNLOAD", "1") != "0" and body.get("uri"):
        log.info("updateFirmware: downloading %s to %s", target, firmware.firmware_dir())
        firmware.download(body["uri"], target, body.get("md5"), _callback_headers(mac_nosep),
                          (CERT_PATH, KEY_PATH))
    wait = max(0.0, FAKE_UPDATE_S - (time.monotonic() - t0))
    log.info("updateFirmware: simulating install of %s for %.0fs", target, wait)
    time.sleep(wait)
    config.pin_fw_version(STATE_DIR, target)
    log.info("updateFirmware: now on %s; simulated reboot (closing ucp4)", target)
    try:
        ws.close()
    except Exception:  # noqa: BLE001 -- already closed is fine
        pass


def _psql_path():
    """psql from DIYKEY_PG_BIN, else PATH, else Homebrew's postgresql@17 (macOS service)."""
    pg_bin = os.environ.get("DIYKEY_PG_BIN")
    if pg_bin:
        return os.path.join(pg_bin, "psql")
    return shutil.which("psql") or "/opt/homebrew/opt/postgresql@17/bin/psql"


def _pg_password_plan(new_password):
    """(argv, stdin) that gives role ``unifi-protect`` the password Protect rotated to, or
    (None, reason) when this host has no reachable PostgreSQL.

    macOS service: DIYKEY_PG_SOCKET=<instance>/pg, psql runs as the login user.
    Linux system PostgreSQL: none needed -- as root psql runs as the postgres OS user over
    /var/run/postgresql, so peer auth lets it ALTER ROLE. DIYKEY_PG_USER overrides the OS
    user (any user that can authenticate as a superuser works).
    """
    sock = os.environ.get("DIYKEY_PG_SOCKET")
    user = os.environ.get("DIYKEY_PG_USER")
    if sys.platform != "darwin" and not sock and os.geteuid() == 0:
        sock = "/var/run/postgresql"
        user = user or "postgres"
    if not sock and not user:
        return None, "no DIYKEY_PG_SOCKET/DIYKEY_PG_USER"
    cmd = [_psql_path(), "-d", "postgres", "-q", "-v", "ON_ERROR_STOP=1"]
    if sock:
        cmd += ["-h", sock]
    runuser = shutil.which("runuser")
    if user and os.geteuid() == 0 and user != "root" and runuser:
        cmd = [runuser, "-u", user, "--"] + cmd
    elif user and user != "root":
        cmd += ["-U", user]
    # The password never goes in argv (it would show in ps): quoted as a SQL literal instead,
    # doubling embedded quotes, with backslash escapes off so it is stored verbatim.
    literal = "'" + new_password.replace("'", "''") + "'"
    sql = ("SET standard_conforming_strings = on;\n"
           f'ALTER ROLE "unifi-protect" PASSWORD {literal};\n')
    return cmd, sql


def _apply_pg_password(new_password):
    """Give the instance's PostgreSQL role the password Protect just rotated to, so its
    first candidate works."""
    if not new_password:
        return
    cmd, payload = _pg_password_plan(new_password)
    if cmd is None:
        log.info("changeUserPassword: %s; PostgreSQL password left as is", payload)
        return
    try:
        r = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("changeUserPassword: psql failed: %r", e)
        return
    if r.returncode:
        log.warning("changeUserPassword: psql exited %d: %s", r.returncode, r.stderr.strip())
    else:
        log.info("changeUserPassword: applied the rotated password to PostgreSQL")
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


def _console_host(fallback_host_port):
    """The adopted console's host: adopt_state hosts[0], else the configured fallback."""
    try:
        with open(ADOPT_STATE_FILE) as f:
            hosts = json.load(f).get("hosts") or []
        if hosts:
            return _split_host_port(hosts[0])[0]
    except (OSError, ValueError, TypeError):
        pass
    return _split_host_port(fallback_host_port)[0] if fallback_host_port else None


def _nl_search_reply(head, body, device_info):
    """Reply body for an nl-search request; empty fields on failure, like the AI Key."""
    action = head.get("action")
    body = body or {}
    try:
        if action == "NL_PARSE":
            reply = nl_search.parse(body.get("querySentence") or "")
            log.info("NL_PARSE %r (model %s) -> tags %s types %s time %s embed %d",
                     body.get("querySentence"), body.get("model"),
                     [t["matchedWord"] for t in reply["keyTags"]], reply["objectTypes"],
                     reply.get("timeTag"), len(reply["txtEmbed"]))
            return reply
        if action == "IMAGE_SEARCH":
            import cv2
            import numpy as np
            resp = requests.get(body.get("imgUri") or "", headers=_callback_headers(
                device_info["mac_nosep"]), cert=(CERT_PATH, KEY_PATH), verify=False, timeout=30)
            img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR) \
                if resp.status_code == 200 else None
            reply = nl_search.image_search(img)
            log.info("IMAGE_SEARCH %s -> embed %d", resp.status_code, len(reply["imgEmbed"]))
            return reply
    except Exception as e:  # noqa: BLE001 -- Protect waits for a reply either way
        log.warning("nl-search %s failed: %r", action, e)
        return {"imgEmbed": []} if action == "IMAGE_SEARCH" else {"keyTags": [], "objectTypes": []}
    log.warning("nl-search: unhandled action %r", action)
    return {}


def _nl_search_loop(fallback_host_port, device_info):
    """Hold the AI Key's second WebSocket (wss://<console>:7443/wss/nl-search/v1) open while
    adopted, so Protect routes Find Anything here (see nl_search.py)."""
    backoff = 5
    while True:
        host = _console_host(fallback_host_port) if config.is_adopted(STATE_DIR) else None
        if not host:
            time.sleep(10)
            continue
        url = f"wss://{host}:{nl_search.PORT}{nl_search.PATH}"
        extra = {}
        if device_info.get("bind_ip") not in (None, "", "0.0.0.0"):
            extra["source_address"] = (device_info["bind_ip"], 0)
        try:
            with connect(url, subprotocols=["ucp4"],
                         additional_headers={"x-ident": device_info["mac_nosep"]},
                         ssl=build_ssl_context(), open_timeout=15, **extra) as ws:
                log.info("nl-search connected to %s", url)
                backoff = 5
                for raw in ws:
                    if isinstance(raw, str):
                        continue
                    records = unpack_all_records(raw)
                    if len(records) < 2:
                        continue
                    head, body = records[0][2], records[1][2]
                    if head.get("type") != "request" or not head.get("id"):
                        continue
                    reply = _nl_search_reply(head, body, device_info)
                    ws.send(pack_message(
                        {"id": head["id"], "type": "response", "action": head.get("action"),
                         "timestamp": int(time.time() * 1000), "error": None, "errorCode": 0},
                        reply))
        except Exception as e:  # noqa: BLE001 -- reconnect on anything
            log.warning("nl-search connection to %s ended: %r -- retrying in %ds", url, e, backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)


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
    global _ai_mode, _latest_fw
    action = head.get("action")
    log.info("<<< controller action=%r head=%s body=%s", action, head, body)

    if action == "getInfo":
        return {
            "mac": device_info["mac_nosep"],
            "type": device_info["type"],
            "version": _fw_version(device_info),
            # Read live so ResetToDefaults/readopt can't leave this stale.
            "adopted": config.is_adopted(STATE_DIR),
            "guid": device_info["guid"],
            "deviceId": device_info["device_id"],
            # The console's host, not ours (used to build upload URLs).
            "connectionHost": device_info.get("console_host") or device_info["ip"],
            "poeType": "802.3bt",  # Protect sets requiredPoeType 802.3bt for an AI Key
            "featureFlags": {**FEATURE_FLAGS, "aiMode": _ai_mode},
        }
    if action == "firmwareState":
        _latest_fw = (body or {}).get("latestFirmwareVersion") or _latest_fw
        return {}
    if action == "updateFirmware":
        target = _version_from_update_uri((body or {}).get("uri")) or _latest_fw
        if not target:
            log.warning("updateFirmware without a target version (uri=%r); ignoring",
                        (body or {}).get("uri"))
            return {}
        if device_info.get("ws") is not None:
            threading.Thread(target=_simulate_update,
                             args=(device_info["ws"], target, body, device_info["mac_nosep"]),
                             daemon=True).start()
        else:
            config.pin_fw_version(STATE_DIR, target)
        return {}
    if action == "networkStatus":
        # Protect reads linkSpeedMbps (and ssid/rssi, wifi only).
        return {"ip": device_info["ip"], "connected": True, "linkSpeedMbps": 1000}
    if action == "changeUserPassword":
        _apply_pg_password((body or {}).get("passwordNew"))
        return {}
    if action == "changeAiInferAgentSettings" and (body or {}).get("modelMode"):
        _ai_mode = body["modelMode"]
        log.info("AI mode set to %s", _ai_mode)
        return {}
    if action == "diskInfo":
        # Protect parses storageSize as a numeric string into featureFlags.storageSize;
        # 128 makes it the UP-AI-KEY-128 variant, anything else the 256 GB OrinNX256.
        return {"storageSize": "256"}
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
            target=_handle_recognize,
            args=(device_info["console_host"], device_info["mac_nosep"], body),
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
                "trackerID": int(roi.get("trackerID", roi.get("trackerId", -1))),  # schema: number
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


def _verify_from_clip(params, mac_nosep):
    """Second Stage Verification on the event's own frames: fetch reqUrl (the export around
    the thumbnails) as MP4, grab each thumbnail time, classify each box. None when there is
    no tagger model or the clip/frames can't be had, so the placeholder goes back instead."""
    tagger = _get_tagger()
    req_url = params.get("reqUrl")
    if not tagger or not req_url or not STATE_DIR:
        return None
    t0 = time.monotonic()
    moments = sorted({e.get("ts") for e in params.get("thumbnailMeta") or [] if e.get("ts")}
                     | set(params.get("thumbnailMs") or []))
    if not moments:
        return None
    start = dict(parse_qsl(urlsplit(req_url).query)).get("start")
    start = int(start) if start and start.isdigit() else moments[0]
    clips_dir = os.path.join(STATE_DIR, "clips")
    clip = clip_fetch.fetch_clip(clip_fetch.mp4_url("", CONSOLE_INTERNAL_API_PORT, req_url),
                                 _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH), clips_dir,
                                 f"{params.get('event') or 'event'}-verify")
    if not clip:
        return None
    frames = clip_fetch.extract_frames(clip, start, moments)
    clip_fetch.prune(clips_dir)
    if not frames:
        return None
    try:
        result = second_verifier.verify(tagger, frames, params)
    except Exception as e:  # noqa: BLE001 -- fall back to the placeholder
        log.warning("verification failed for event %s: %r", params.get("event"), e)
        return None
    pre_ms = int((time.monotonic() - t0) * 1000) - result["inferMs"]
    result["preProcessMs"] = max(0, pre_ms)
    result["timeElapsedMs"] = result["preProcessMs"] + result["inferMs"]
    return result


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
        result = _verify_from_clip(params, mac_nosep) or _second_verifier_result(params)
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


_tagger = None
_tagger_lock = threading.Lock()


def _get_tagger():
    """The RAM tagger for DIYKEY_RAM_MODEL, loaded once; None when unset or broken."""
    global _tagger
    path = os.environ.get("DIYKEY_RAM_MODEL")
    if not path:
        return None
    with _tagger_lock:
        if _tagger is None:
            try:
                _tagger = ram_tagger.RamTagger(path)
            except Exception as e:  # noqa: BLE001 -- a bad model must not stop results
                log.warning("RAM tagger for %s failed to load: %r", path, e)
                _tagger = False
        return _tagger or None


def _handle_recognize(console_host, mac_nosep, body):
    """Fetch the task's clip, tag its key-moment and thumbnail frames, and POST
    the result. Every step is best effort: on a failure the placeholder goes back."""
    camera_id, event_id = body.get("camera"), body.get("event")
    result = None
    req_url = body.get("reqUrl")
    if body.get("ramType") in ("image", "multipleImages") and STATE_DIR:
        # Retroactive processing (and image RAM): cropped JPEGs by imageId, no clip.
        result = _recognize_images(console_host, mac_nosep, body)
    elif req_url and STATE_DIR:
        t0 = time.monotonic()
        clips_dir = os.path.join(STATE_DIR, "clips")
        url = clip_fetch.mp4_url(console_host, CONSOLE_INTERNAL_API_PORT, req_url)
        clip = clip_fetch.fetch_clip(url, _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH),
                                     clips_dir, event_id)
        if clip and body.get("start") is not None:
            moments = sorted(set(body.get("keyMoments") or []) | set(body.get("thumbnailMs") or []))
            frames = clip_fetch.extract_frames(clip, body["start"], moments)
            log.info("event %s: %d/%d frames", event_id, len(frames), len(moments))
            tagger = _get_tagger()
            if tagger and frames:
                try:
                    result = ram_tagger.build_result(tagger, camera_id, event_id, frames,
                                                     pre_ms=(time.monotonic() - t0) * 1000,
                                                     objects=_video_objects(body, frames))
                    log.info("event %s: %s", event_id, result["description"])
                except Exception as e:  # noqa: BLE001
                    log.warning("RAM tagging failed for event %s: %r", event_id, e)
        clip_fetch.prune(clips_dir)
    _post_recognize_result(console_host, mac_nosep, body.get("resUrl"), camera_id, event_id,
                           result)


def _video_objects(body, frames):
    """Tracked objects of a video RAM task, cropped out of the frame at their thumbnail time."""
    import cv2
    objects = []
    for entry in body.get("thumbnailMeta") or []:
        ts = entry.get("ts")
        img = cv2.imread(frames[ts]) if ts in frames else None
        for roi in entry.get("roi") or []:
            tracker = roi.get("trackerID", roi.get("trackerId"))
            if img is None or tracker is None or not roi.get("coord"):
                continue
            objects.append({"trackerID": tracker, "keyMomentMs": ts,
                            "img": ram_tagger.crop_box(img, roi["coord"]), "hint": roi})
    return objects


def _recognize_images(console_host, mac_nosep, body):
    """RAM result for an image task: ramType "image" (reqUrl + keyMoment) or "multipleImages"
    (images[] with their own reqUrl and timestamps). None when nothing could be tagged."""
    tagger = _get_tagger()
    if not tagger:
        return None
    t0 = time.monotonic()
    event_id = body.get("event") or "event"
    if body.get("ramType") == "image":
        entries = [body]
    else:
        entries = list(body.get("images") or [])
    items = [(e.get("reqUrl"), e.get("keyMoment") or body.get("keyMoment") or body.get("start"),
              e.get("imageId"), e) for e in entries]
    clips_dir = os.path.join(STATE_DIR, "clips")
    frames, hints, objects = {}, {}, []
    for n, (url, ms, image_id, entry) in enumerate(items):
        if not url or ms is None:
            continue
        url = url if url.startswith("http") else \
            f"https://{console_host}:{CONSOLE_INTERNAL_API_PORT}{url}"
        path = os.path.join(clips_dir, f"{event_id}-images", f"{image_id or n}.jpg")
        if clip_fetch.fetch_image(url, _callback_headers(mac_nosep), (CERT_PATH, KEY_PATH), path):
            frames[int(ms)] = path
            if entry.get("objectType") or entry.get("attributes"):
                hints[int(ms)] = entry
            tracker = entry.get("trackerId", entry.get("trackerID"))
            if tracker is not None:
                import cv2
                objects.append({"trackerID": tracker, "keyMomentMs": int(ms),
                                "img": cv2.imread(path), "hint": entry})
    clip_fetch.prune(clips_dir)
    log.info("event %s: %d/%d images (%s)", event_id, len(frames), len(items), body.get("ramType"))
    if not frames:
        return None
    try:
        result = ram_tagger.build_result(tagger, body.get("camera"), body.get("event"), frames,
                                         pre_ms=(time.monotonic() - t0) * 1000, hints=hints,
                                         objects=objects)
    except Exception as e:  # noqa: BLE001
        log.warning("RAM image tagging failed for event %s: %r", event_id, e)
        return None
    log.info("event %s: %s", event_id, result["description"])
    return result


def _post_recognize_result(console_host, mac_nosep, res_url, camera_id, event_id, result=None):
    """POST our RAM/recognize-anything result to resUrl.

    CONFIRMED from the real firmware's syswrapper.sh: this is multipart/form-data
    with the JSON under form field ``ram`` (optionally with per-tracker JPEG
    fields), not a plain JSON POST. `result` is a ramInferenceSchema dict from
    ram_tagger.build_result, or None for the placeholder.
    """
    if not res_url:
        log.warning("recognizeKeyFrames task had no resUrl, nothing to post")
        return
    url = res_url if res_url.startswith("http") else \
        f"https://{console_host}:{CONSOLE_INTERNAL_API_PORT}{res_url}"
    ram = result or {
        "cameraId": camera_id,
        "eventId": event_id,
        "status": "success",
        "keyMomentsTags": [],
    }
    files = {"ram": ("ram.json", json.dumps(ram), "application/json")}
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
        "X-Sysid": device_info["sysid"],
        "X-Adopted": "true" if adopted else "false",
        "X-Ip": device_info["ip"],
        "X-Device-Id": device_info["device_id"],
        "X-Version": _fw_version(device_info),
        "X-Guid": device_info["guid"],
    }
    if send_token_header:
        headers["X-Token"] = token

    log.info("connecting to %s with headers=%s", url, headers)
    ctx = build_ssl_context()

    # Leave from the bound address too, so a second IP on a shared host is the
    # one the console sees (websockets passes this to socket.create_connection).
    extra = {}
    if device_info.get("bind_ip") not in (None, "", "0.0.0.0"):
        extra["source_address"] = (device_info["bind_ip"], 0)

    with connect(url, subprotocols=["ucp4"], additional_headers=headers,
                 ssl=ctx, open_timeout=15, **extra) as ws:
        log.info("WSS CONNECTED (subprotocol=%s)", ws.subprotocol)
        device_info["ws"] = ws  # this connection's copy; updateFirmware closes it to "reboot"

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
        "bind_ip": args.bind_ip,
        "sysid": f"0x{config.sysid_int(cfg):04x}",
    }

    if clip_embed.enabled():
        # Load CLIP now: the first query would otherwise wait ~10 s for it.
        threading.Thread(target=clip_embed.text_embed, args=("warm up",), daemon=True).start()
    if os.environ.get("DIYKEY_NL_SEARCH", "1") != "0":
        threading.Thread(target=_nl_search_loop, args=(args.fallback_host_port, device_info),
                         daemon=True).start()

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
