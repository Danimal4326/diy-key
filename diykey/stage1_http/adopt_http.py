#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""HTTPS control API on :8080, standing in for the real AI Key's local httpd.

CONFIRMED from a live capture of the real UNVR's own errors.log and from the
protocol writeup (``../../protocol.md`` §2): the console polls ``GET /api/info``
about every 60 s and sends ``POST /api/adopt`` with a flat body carrying
``hosts``/``token``/``protocol``/``consoleId``. Other ``/api/<action>`` posts
(``readopt``, ``support``, ``snapshot``, ``locate``) are acknowledged with 200.

An adopt POST is persisted atomically to ``<state-dir>/adopt_state.json`` and
read back live by every stage, so ``adopted`` never goes stale in memory.
"""
import argparse
import http.server
import json
import logging
import os
import ssl
import subprocess
import sys
import time
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core import config  # noqa: E402

log = logging.getLogger("adopt-http")

REQUEST_TIMEOUT_S = 10
MAX_BODY_BYTES = 1 << 20

STATE_DIR = None
_PROCESS_START = time.time()


def ensure_cert(state_dir=None):
    """Generate the self-signed EC P-256 device cert once (the console pins its
    fingerprint at adoption, so it must survive restarts). The same keypair is
    the client identity for ucp4."""
    paths = config.state_paths(state_dir)
    cert, key = paths["cert"], paths["key"]
    if os.path.exists(cert) and os.path.exists(key):
        return cert, key
    log.info("generating self-signed EC P-256 device cert (controller does not validate it)")
    subprocess.run([
        "openssl", "ecparam", "-out", key, "-name", "prime256v1", "-genkey", "-noout",
    ], check=True)
    os.chmod(key, 0o600)
    subprocess.run([
        "openssl", "req", "-new", "-x509", "-sha256", "-key", key, "-out", cert,
        "-days", "36500", "-subj", "/O=diykey/CN=diykey",
    ], check=True)
    return cert, key


def persist_adopt(state_dir, payload):
    """Validate and atomically persist an adopt payload. Returns False and
    leaves existing state untouched when the body is unusable."""
    if not isinstance(payload, dict) or not (
            payload.get("hosts") or payload.get("token")):
        log.warning("ignoring adopt payload without hosts/token")
        return False
    log.info("=" * 70)
    log.info("ADOPT PAYLOAD RECEIVED")
    log.info("  hosts:      %s", payload.get("hosts"))
    log.info("  token:      %s", payload.get("token"))
    log.info("  protocol:   %s", payload.get("protocol"))
    log.info("  consoleId:  %s", payload.get("consoleId"))
    log.info("  consoleName:%s", payload.get("consoleName"))
    log.info("  username:   %s", payload.get("username"))
    log.info("=" * 70)
    config.write_adopt_state(state_dir, payload)
    return True


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "diykey-http/0.2"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        """Body bytes, or None after a 413 was sent."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > MAX_BODY_BYTES:
            log.warning("refusing %d-byte body from %s", length, self.client_address)
            self._json(413, {"statusCode": 413})
            return None
        if not length:
            return b""
        return self.rfile.read(length)

    def _info_payload(self):
        cfg = self.server.diykey_config
        adopted = config.is_adopted(STATE_DIR)
        return {
            "mac": cfg["mac"],
            "type": cfg["platform"],
            "sysid": f"0x{cfg['sysid']:04x}",
            "name": cfg["hostname"],
            "model": cfg["platform"],
            "firmwareVersion": config.effective_fw_version(cfg, STATE_DIR),
            "adopted": adopted,
            "isAdopted": adopted,
            "guid": cfg["guid"],
            "deviceId": cfg["device_id"],
            "anonymousDeviceId": cfg["device_id"],
            "ip": cfg["ip"],
            "uptime": int(time.time() - _PROCESS_START),
            "connectionHost": cfg["ip"],
            "hardwareRevision": None,
        }

    def do_GET(self):
        log.info("GET %s from %s", self.path, self.client_address)
        path = self.path.rstrip("/")
        if path in ("/api/info", "/info"):
            self._json(200, self._info_payload())
        elif path in ("/api/support", "/support"):
            self._json(200, {"statusCode": 200})
        else:
            log.warning("unhandled GET path %s -- replying 200 anyway", self.path)
            self._json(200, {"statusCode": 200})

    def do_POST(self):
        body = self._read_body()
        if body is None:
            return
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {"_raw": body.decode(errors="replace")}
        log.info("POST %s from %s body=%s", self.path, self.client_address,
                 json.dumps(payload))

        path = self.path.rstrip("/")
        if path in ("/api/adopt", "/adopt"):
            if not persist_adopt(STATE_DIR, payload):
                self._json(400, {"statusCode": 400, "error": "invalid adopt payload"})
                return
            self._json(200, {"statusCode": 200})
        elif path in ("/api/readopt", "/readopt"):
            # A real un-adopt clears the token; leaving it would make discovery
            # keep advertising "managed" and the next WSS connect stale.
            config.clear_adopt_state(STATE_DIR)
            log.info("readopt: cleared local adopt state")
            self._json(200, {"statusCode": 200})
        elif path in ("/api/info", "/info"):
            # Protect 7.3 POSTs here with {"username","password"}; no real auth.
            self._json(200, self._info_payload())
        else:
            log.warning("unhandled POST path %s -- replying 200 anyway", self.path)
            self._json(200, {"statusCode": 200})

    def log_message(self, fmt, *args):
        pass  # quieted; we log explicitly above


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    ssl_context = None

    def process_request_thread(self, request, client_address):
        """TLS-wrap in the worker thread, not in serve_forever's accept loop.

        Wrapping the listening socket made one stalled ClientHello block every
        subsequent connection.
        """
        try:
            request.settimeout(REQUEST_TIMEOUT_S)
            tls = self.ssl_context.wrap_socket(request, server_side=True)
        except (OSError, ssl.SSLError) as exc:
            log.warning("TLS handshake from %s failed: %s", client_address, exc)
            self.shutdown_request(request)
            return
        try:
            self.finish_request(tls, client_address)
        except Exception:
            traceback.print_exc()
            self.handle_error(tls, client_address)
        finally:
            self.shutdown_request(tls)


def main():
    cfg = config.load_config_and_logging(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    config.add_common_flags(ap)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--iface", default=None,
                    help="interface to derive identity/IP from; defaults to cfg "
                         "[network] iface, else the default-route interface")
    ap.add_argument("--bind-ip", default="0.0.0.0",
                    help="address to listen on; run_all.py passes the instance's own "
                         "IP so several instances can share :8080")
    ap.add_argument("--mac", default=None)
    ap.add_argument("--hostname", default=cfg["hostname"])
    ap.add_argument("--platform", default=cfg["platform"],
                    help="canonical Protect type (ucp4 X-Type / /api/info)")
    ap.add_argument("--sysid", type=lambda s: int(s, 0), default=config.sysid_int(cfg))
    ap.add_argument("--fw-version", default=cfg["fw_version"])
    ap.add_argument("--device-id", default=cfg["device_id"])
    ap.add_argument("--guid", default=cfg["guid"])
    ap.add_argument("--state-dir", default=None,
                    help="where adopt_state.json/server.crt/server.key live -- defaults "
                         "to stage1_http/; set this to isolate another instance")
    args = ap.parse_args()
    args.iface = config.resolve_iface(args.iface, cfg)

    global STATE_DIR
    STATE_DIR = config.resolve_state_dir(args.state_dir)

    cert, key = ensure_cert(STATE_DIR)
    ip = config.advertised_ip(config.resolve_ip(args.iface, cfg, args.bind_ip))

    server = Server((args.bind_ip, args.port), Handler)
    server.diykey_config = {
        "mac": config.resolve_mac(args.iface, args.mac, cfg),
        "platform": args.platform,
        "sysid": args.sysid,
        "hostname": args.hostname,
        "fw_version": args.fw_version,
        "device_id": args.device_id,
        "guid": args.guid,
        "ip": ip,
    }

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    server.ssl_context = ctx

    log.info("HTTPS control API listening on %s:%d (mac=%s platform=%s state=%s)",
             args.bind_ip, args.port, server.diykey_config["mac"], args.platform, STATE_DIR)
    server.serve_forever()


if __name__ == "__main__":
    main()
