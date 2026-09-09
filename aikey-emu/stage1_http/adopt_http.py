#!/usr/bin/env python3
"""HTTPS control API on :8080, standing in for the real AI Key's local httpd.

CONFIRMED (not guessed) from a live capture of the real UNVR's own
errors.log while attempting to adopt this emulator:
  - The console polls `GET .../info` roughly every 60s, both before and
    after clicking Adopt.
  - Clicking Adopt triggers `POST .../adopt`.
  - Both go to port 8080 on the device's IP (log line: "connect ECONNREFUSED
    <device-ip>:8080" for both actions) -- nothing was listening there
    before this, which is exactly why adoption failed with a bare 500 and
    zero detail client-side.
  - This matches ../PROTOCOL_SPEC.md's CONFIRMED chime protocol almost
    exactly (also HTTPS :8080, also GET .../info + POST .../adopt, also
    server cert NOT validated by the controller) -- so this implementation
    follows that spec's shape for the parts AI Key hasn't diverged from,
    and will get corrected once we see a real adopt POST body/response
    requirement from the controller's next log lines.

Exact JSON response shapes for /api/info and /api/adopt are NOT yet
confirmed for aiprocessors specifically (chime's shapes are confirmed;
these are adapted from it plus the real `aiprocessors` bootstrap record
fields seen live) -- if the controller logs a parse error or silently
doesn't progress past this point, check errors.log / aiprocessors.log on
the controller next.
"""
import argparse
import datetime
import http.server
import json
import logging
import os
import ssl
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("adopt-http")

HERE = os.path.dirname(os.path.abspath(__file__))
CERT_PATH = os.path.join(HERE, "server.crt")
KEY_PATH = os.path.join(HERE, "server.key")
ADOPT_STATE_FILE = os.path.join(HERE, "adopt_state.json")


def ensure_cert():
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        return
    log.info("generating self-signed EC P-256 device cert (controller does not validate it)")
    subprocess.run([
        "openssl", "ecparam", "-out", KEY_PATH, "-name", "prime256v1", "-genkey", "-noout",
    ], check=True)
    subprocess.run([
        "openssl", "req", "-new", "-x509", "-sha256", "-key", KEY_PATH, "-out", CERT_PATH,
        "-days", "36500", "-subj", "/O=aikey-emu/CN=aikey-emu",
    ], check=True)


def get_mac(iface: str) -> str:
    try:
        with open(f"/sys/class/net/{iface}/address") as f:
            return f.read().strip().upper().replace(":", "")
    except OSError:
        return "000000000000"


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "aikey-emu-http/0.1"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return b""
        return self.rfile.read(length)

    def _info_payload(self):
        cfg = self.server.aikey_config
        return {
            "mac": cfg["mac"],
            "type": cfg["platform"],
            "sysid": f"0x{cfg['sysid']:04x}",
            "name": cfg["hostname"],
            "model": cfg["platform"],
            "firmwareVersion": cfg["fw_version"],
            "adopted": cfg["adopted"],
            "isAdopted": cfg["adopted"],
            "guid": cfg["guid"],
            "deviceId": cfg["device_id"],
            "anonymousDeviceId": cfg["device_id"],
            "ip": cfg["ip"],
            "uptime": 0,
            "connectionHost": cfg["ip"],
            "hardwareRevision": None,
        }

    def do_GET(self):
        log.info("GET %s from %s", self.path, self.client_address)
        if self.path.rstrip("/") in ("/api/info", "/info"):
            self._json(200, self._info_payload())
        elif self.path.rstrip("/") in ("/api/support", "/support"):
            self._json(200, {"statusCode": 200})
        else:
            log.warning("unhandled GET path %s -- replying 200 anyway", self.path)
            self._json(200, {"statusCode": 200})

    def do_POST(self):
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {"_raw": body.decode(errors="replace")}
        log.info("POST %s from %s body=%s", self.path, self.client_address,
                  json.dumps(payload))

        path = self.path.rstrip("/")
        if path in ("/api/adopt", "/adopt"):
            self.server.aikey_config["adopted"] = True
            self.server.aikey_config["last_adopt_payload"] = payload
            log.info("=" * 70)
            log.info("ADOPT PAYLOAD RECEIVED")
            log.info("  hosts:      %s", payload.get("hosts"))
            log.info("  token:      %s", payload.get("token"))
            log.info("  protocol:   %s", payload.get("protocol"))
            log.info("  consoleId:  %s", payload.get("consoleId"))
            log.info("  consoleName:%s", payload.get("consoleName"))
            log.info("  username:   %s", payload.get("username"))
            log.info("=" * 70)
            with open(ADOPT_STATE_FILE, "w") as f:
                json.dump(payload, f)
            self._json(200, {"statusCode": 200})
        elif path in ("/api/readopt", "/readopt"):
            self._json(200, {"statusCode": 200})
        else:
            log.warning("unhandled POST path %s -- replying 200 anyway", self.path)
            self._json(200, {"statusCode": 200})

    def log_message(self, fmt, *args):
        pass  # quieted; we log explicitly above


class Server(http.server.ThreadingHTTPServer):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--hostname", default="aikey-emu")
    ap.add_argument("--platform", default="UP-AI-KEY")
    ap.add_argument("--sysid", type=lambda s: int(s, 0), default=0xa5f0)
    ap.add_argument("--fw-version", default="2.2.6")
    ap.add_argument("--device-id", default="00000000-0000-4000-8000-000000000001")
    ap.add_argument("--guid", default="00000000-0000-4000-8000-000000000002")
    args = ap.parse_args()

    ensure_cert()

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "0.0.0.0"
    finally:
        s.close()

    server = Server(("0.0.0.0", args.port), Handler)
    server.aikey_config = {
        "mac": get_mac(args.iface),
        "platform": args.platform,
        "sysid": args.sysid,
        "hostname": args.hostname,
        "fw_version": args.fw_version,
        "device_id": args.device_id,
        "guid": args.guid,
        "ip": ip,
        "adopted": False,
        "last_adopt_payload": None,
    }

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT_PATH, KEY_PATH)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)

    log.info("HTTPS control API listening on 0.0.0.0:%d (mac=%s platform=%s)",
              args.port, server.aikey_config["mac"], args.platform)
    server.serve_forever()


if __name__ == "__main__":
    main()
