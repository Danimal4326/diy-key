# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Watch an AI Key's Protect record and its journald log side by side.

Hosts and credentials come from the environment; nothing is hardcoded:
  UNVR_HOST (required), UNVR_USER (default root), UNVR_PASS (required)
  DIYKEY_INSTANCE (default main -> diy-key-instance-main.service)
  AIPROC_ID (optional; all aiprocessors are watched when unset)
  ORANGEPI_LOG (override the journal command entirely)
"""
import http.cookiejar
import json
import os
import ssl
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_ssh  # noqa: E402

UNVR_HOST = os.environ.get("UNVR_HOST")
if not UNVR_HOST:
    sys.exit("UNVR_HOST is unset")
UNVR_USER = os.environ.get("UNVR_USER", "root")
if not os.environ.get("UNVR_PASS"):
    sys.exit("UNVR_PASS is unset")

AIPROCESSORS_API = os.environ.get("AIPROCESSORS_API", "/proxy/protect/api/aiprocessors")
AIPROC_ID = os.environ.get("AIPROC_ID")
DIYKEY_INSTANCE = os.environ.get("DIYKEY_INSTANCE", "main")
PI_LOG = os.environ.get(
    "ORANGEPI_LOG", f"journalctl -u diy-key-instance-{DIYKEY_INSTANCE} -n 200 --no-pager")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj),
                                     urllib.request.HTTPSHandler(context=ctx))
body = json.dumps({"username": UNVR_USER, "password": os.environ["UNVR_PASS"]}).encode()
req = urllib.request.Request(f"https://{UNVR_HOST}/api/auth/login", data=body, method="POST",
                             headers={"Content-Type": "application/json"})
resp = opener.open(req, timeout=10)
csrf = resp.headers.get("X-Updated-Csrf-Token")

aiprocs_url = f"https://{UNVR_HOST}{AIPROCESSORS_API}" + (f"/{AIPROC_ID}" if AIPROC_ID else "")

last_state = None
last_log_line_count = 0

print(f"watching {aiprocs_url} ... (ctrl-c to stop)", flush=True)
start = time.time()
while time.time() - start < 90:
    try:
        req = urllib.request.Request(aiprocs_url, headers={"X-Csrf-Token": csrf})
        r = opener.open(req, timeout=5)
        d = json.loads(r.read())
        entries = d if isinstance(d, list) else [d]
        snapshot = [{k: e.get(k) for k in [
            "state", "canAdopt", "isAttemptingToConnect", "isAdopting", "isAdopted",
            "isProvisioned", "connectionHost", "lastSeen", "connectedSince", "name",
            "mac", "modelKey"]}
            for e in entries]
        if snapshot != last_state:
            print(f"[{time.strftime('%H:%M:%S')}] NVR state change: {snapshot}", flush=True)
            last_state = snapshot
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] NVR poll error: {e}", flush=True)

    try:
        rc, out, err = pi_ssh.run(PI_LOG, timeout=10)
        lines = out.splitlines()
        if len(lines) > last_log_line_count:
            for line in lines[last_log_line_count:]:
                print(f"[PI LOG] {line}", flush=True)
            last_log_line_count = len(lines)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] PI poll error: {e}", flush=True)

    time.sleep(1)
print("done watching")
