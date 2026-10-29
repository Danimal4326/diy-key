# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep a copy of the firmware Protect offers, for later investigation.

A (simulated) updateFirmware carries a console URL such as
``https://<console>:7444/internal/update?platform=OrinNX256&product=AI-Key&
updateType=firmware&version=2.2.8&mac=...`` plus the file's md5. The console
serves it to our client cert like the other /internal endpoints. Files go to
DIYKEY_FW_DIR (default ~/.diy-key/firmware, shared by instances and outside
any repo -- vendor firmware is never committed), named
``<product>-<platform>-<version>.tar`` with a ``.json`` of where it came from.
"""
import datetime
import hashlib
import json
import logging
import os
import re
from urllib.parse import parse_qsl, urlsplit

import requests

log = logging.getLogger("diykey.firmware")

TIMEOUT_S = 60
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def firmware_dir():
    return os.path.expanduser(os.environ.get("DIYKEY_FW_DIR") or "~/.diy-key/firmware")


def target_name(uri, version):
    """<product>-<platform>-<version>.tar from the update URL's fields."""
    q = dict(parse_qsl(urlsplit(uri).query)) if uri else {}
    parts = [q.get("product") or "AI-Key", q.get("platform") or "unknown", version or "unknown"]
    return "-".join(_SAFE.sub("_", p) for p in parts) + ".tar"


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(uri, version, md5, headers, cert, out_dir=None):
    """Fetch the update to out_dir, check the md5, write the sidecar; returns the path or
    None. A file already there with the right md5 is kept as is."""
    out_dir = out_dir or firmware_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, target_name(uri, version))
    if os.path.exists(path) and (not md5 or _md5(path) == md5):
        log.info("firmware %s already stored at %s", version, path)
        return path
    tmp = path + ".part"
    h = hashlib.md5()
    try:
        with requests.get(uri, headers=headers, cert=cert, verify=False, stream=True,
                          timeout=TIMEOUT_S) as resp:
            if resp.status_code != 200:
                log.warning("firmware fetch -> %s: %s", resp.status_code, resp.text[:300])
                return None
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
    except (requests.RequestException, OSError) as e:
        log.warning("firmware fetch failed: %r", e)
        _remove(tmp)
        return None
    got = h.hexdigest()
    if md5 and got != md5:
        log.warning("firmware %s md5 mismatch (got %s, want %s); not kept", version, got, md5)
        _remove(tmp)
        return None
    os.replace(tmp, path)
    q = dict(parse_qsl(urlsplit(uri).query))
    meta = {"version": version, "md5": got, "size": os.path.getsize(path),
            "product": q.get("product"), "platform": q.get("platform"),
            "source": f"{urlsplit(uri).scheme}://{urlsplit(uri).netloc}{urlsplit(uri).path}",
            "downloaded": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("firmware %s stored at %s (%d bytes, md5 ok)", version, path, meta["size"])
    return path


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
