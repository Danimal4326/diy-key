#!/usr/bin/env python3
"""Model-serving HTTP endpoint on :7788, matching the real ui-infer-gateway.

Confirmed live and from the firmware's ui-infer-gateway package. Returns the real
`second_verifier_mlabel` result shape with placeholder values.
"""
import argparse
import http.server
import json
import logging
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core import config  # noqa: E402

log = logging.getLogger("model-server")


def _second_verifier_result(params: dict) -> dict:
    """Placeholder result in the CONFIRMED shape (see module docstring),
    keeping every claimed ROI's detector class as-is (modelRunStatus
    "skipped" -- no real classification happens here yet)."""
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


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "diykey-modelserver/0.1"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"_raw": raw.decode(errors="replace")}

        log.info("POST %s from %s body=%s", self.path, self.client_address,
                  json.dumps(payload))

        # Path is /v1/models/{model_name}/inference per the real gateway.
        parts = self.path.strip("/").split("/")
        model_name = parts[2] if len(parts) >= 4 and parts[0] == "v1" and parts[1] == "models" else "unknown"
        action = payload.get("action", "default")
        params = payload.get("params", {})

        if model_name == "second_verifier_mlabel":
            result = _second_verifier_result(params)
        else:
            result = {}

        response = {
            "model": model_name,
            "action": action,
            "inference_time_ms": 0.0,
            "result": result,
        }
        log.info("responding %s", response)
        self._json(200, response)

    def do_GET(self):
        log.info("GET %s from %s", self.path, self.client_address)
        self._json(200, {"status": "ok"})

    def log_message(self, fmt, *args):
        pass


def main():
    config.load_config_and_logging(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    config.add_common_flags(ap)
    ap.add_argument("--port", type=int, default=7788)
    ap.add_argument("--bind-ip", default="0.0.0.0",
                    help="address to listen on; run_all.py passes the instance's own "
                         "IP so several instances can share :7788")
    args = ap.parse_args()

    server = http.server.ThreadingHTTPServer((args.bind_ip, args.port), Handler)
    log.info("model-server stub listening on %s:%d", args.bind_ip, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
