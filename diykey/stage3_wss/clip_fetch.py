# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch a task's video clip from the console and pull frames out of it.

recognizeKeyFrames (and reverification) hand us a console-relative reqUrl,
``/internal/aiprocessors/video/export?...&format=ubv``. The console serves the
same export as plain MP4 when asked for ``format=mp4``, which ffmpeg reads; the
fetch authenticates like the result callbacks (our client cert + x-ident).

Clips land in ``<state>/clips/<event>.mp4`` with key-moment JPEGs in
``<state>/clips/<event>/<ts>.jpg``; only the newest KEEP_EVENTS are kept.
"""
import logging
import os
import re
import shutil
import subprocess

import requests

log = logging.getLogger("diykey.clips")

KEEP_EVENTS = 20
FETCH_TIMEOUT_S = 60
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def mp4_url(console_host, port, req_url):
    """Absolute URL for reqUrl, asking for MP4 instead of UBV."""
    url = req_url if req_url.startswith("http") else f"https://{console_host}:{port}{req_url}"
    if "format=" in url:
        return re.sub(r"format=[^&]*", "format=mp4", url)
    return url + ("&" if "?" in url else "?") + "format=mp4"


def fetch_clip(url, headers, cert, clips_dir, event_id):
    """Stream the export to <clips_dir>/<event>.mp4; returns the path, or None."""
    os.makedirs(clips_dir, exist_ok=True)
    path = os.path.join(clips_dir, f"{_SAFE.sub('_', event_id or 'event')}.mp4")
    tmp = path + ".part"
    try:
        with requests.get(url, headers=headers, cert=cert, verify=False, stream=True,
                          timeout=FETCH_TIMEOUT_S) as resp:
            if resp.status_code != 200:
                log.warning("clip fetch %s -> %s: %s", url, resp.status_code, resp.text[:300])
                return None
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(1 << 16):
                    f.write(chunk)
        os.replace(tmp, path)
    except (requests.RequestException, OSError) as e:
        log.warning("clip fetch %s failed: %r", url, e)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    log.info("clip for event %s: %s (%d bytes)", event_id, path, os.path.getsize(path))
    return path


def extract_frames(clip_path, start_ms, moments_ms, ffmpeg="ffmpeg"):
    """One JPEG per key moment (epoch ms), seeking from the clip's start (epoch ms).
    Returns {moment_ms: jpeg_path} for the frames that came out."""
    out_dir = os.path.splitext(clip_path)[0]
    os.makedirs(out_dir, exist_ok=True)
    frames = {}
    for ts in moments_ms:
        offset = max(0.0, (int(ts) - int(start_ms)) / 1000.0)
        jpg = os.path.join(out_dir, f"{int(ts)}.jpg")
        try:
            r = subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{offset:.3f}", "-i", clip_path,
                                "-frames:v", "1", "-q:v", "3", jpg],
                               capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as e:
            log.warning("ffmpeg failed for %s @%.3fs: %r", clip_path, offset, e)
            continue
        if r.returncode == 0 and os.path.exists(jpg):
            frames[int(ts)] = jpg
        else:
            log.warning("no frame from %s @%.3fs: %s", clip_path, offset, r.stderr.strip()[:300])
    return frames


def prune(clips_dir, keep=KEEP_EVENTS):
    """Keep the newest `keep` clips (and their frame folders)."""
    try:
        clips = sorted((e for e in os.scandir(clips_dir) if e.name.endswith(".mp4")),
                       key=lambda e: e.stat().st_mtime, reverse=True)
    except OSError:
        return
    for e in clips[keep:]:
        try:
            os.remove(e.path)
            shutil.rmtree(os.path.splitext(e.path)[0], ignore_errors=True)
        except OSError as err:
            log.warning("could not prune %s: %r", e.path, err)
    # Image tasks (retroactive processing) leave <event>-images/ folders instead of clips.
    try:
        image_dirs = sorted((e for e in os.scandir(clips_dir)
                             if e.is_dir() and e.name.endswith("-images")),
                            key=lambda e: e.stat().st_mtime, reverse=True)
    except OSError:
        return
    for e in image_dirs[keep:]:
        shutil.rmtree(e.path, ignore_errors=True)


def fetch_image(url, headers, cert, out_path):
    """Fetch one JPEG (e.g. /internal/aiprocessors/image/<id>) to out_path; path or None."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        resp = requests.get(url, headers=headers, cert=cert, verify=False, timeout=FETCH_TIMEOUT_S)
    except requests.RequestException as e:
        log.warning("image fetch %s failed: %r", url, e)
        return None
    if resp.status_code != 200 or not resp.content:
        log.warning("image fetch %s -> %s: %s", url, resp.status_code, resp.text[:300])
        return None
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path
