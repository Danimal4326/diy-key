# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch a task's video clip from the console and pull frames out of it.

recognizeKeyFrames (and reverification) hand us a console-relative reqUrl,
``/internal/aiprocessors/video/export?...&format=ubv``. The console serves the
same export as plain MP4 when asked for ``format=mp4``, which ffmpeg reads; the
fetch authenticates like the result callbacks (our client cert + x-ident).

The export does not start exactly at the request's ``start``: live 2026-11-08 07:54 on
Backyard the 31.3 s window came back as a 35.5 s clip, and frames seeked from ``start`` were
4-5 s early by the camera's OSD clock (the face crops held a deck railing). The firmware takes
the clip's real start from the ``x-start-timestamp`` response header (syswrapper.sh 2.2.8
~2590 "modify start by x-start-timestamp"), so fetch_clip saves it next to the clip as
``<event>.mp4.start`` and extract_frames seeks from it. Without the header the clip is taken
to end at the request's ``end`` (start = end - duration).

Clips land in ``<state>/clips/<event>.mp4`` with key-moment JPEGs in
``<state>/clips/<event>/<ts>.jpg``; only the newest KEEP_EVENTS are kept.
"""
import logging
import os
import re
import shutil
import subprocess
import time

import requests
from urllib.parse import parse_qsl, urlsplit

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
            hdr_start = resp.headers.get("x-start-timestamp") or resp.headers.get("x-timestamp")
        os.replace(tmp, path)
    except (requests.RequestException, OSError) as e:
        log.warning("clip fetch %s failed: %r", url, e)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    real, how = clip_start(url, hdr_start, path)
    if real is not None:
        with open(path + ".start", "w") as f:
            f.write(str(real))
    log.info("clip for event %s: %s (%d bytes), starts %s (%s)", event_id, path,
             os.path.getsize(path), real, how)
    return path


def _probe_duration_ms(clip_path, ffprobe="ffprobe"):
    try:
        r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", clip_path], capture_output=True, text=True, timeout=30)
        return int(float(r.stdout.strip()) * 1000)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def clip_start(url, header_value, clip_path, probe=_probe_duration_ms):
    """(real clip start in epoch ms, source) from the response header, else end - duration."""
    try:
        if header_value and int(header_value) > 0:
            return int(header_value), "x-start-timestamp"
    except ValueError:
        pass
    q = dict(parse_qsl(urlsplit(url).query))
    try:
        start, end = int(q["start"]), int(q["end"])
    except (KeyError, ValueError):
        return None, "unknown"
    dur = probe(clip_path)
    if not dur or dur <= end - start:
        return start, "request start"
    return end - dur, "end - duration"


def real_start(clip_path, start_ms):
    """The clip's saved real start, else the caller's start."""
    try:
        with open(clip_path + ".start") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return start_ms


def extract_frames(clip_path, start_ms, moments_ms, ffmpeg="ffmpeg"):
    """One JPEG per key moment (epoch ms), seeking from the clip's start (epoch ms).
    Returns {moment_ms: jpeg_path} for the frames that came out."""
    out_dir = os.path.splitext(clip_path)[0]
    os.makedirs(out_dir, exist_ok=True)
    start_ms = real_start(clip_path, start_ms)
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
        elif _last_frame(clip_path, jpg, ffmpeg):
            log.info("no frame from %s @%.3fs; used the clip's last frame", clip_path, offset)
            frames[int(ts)] = jpg
        else:
            log.warning("no frame from %s @%.3fs: %s", clip_path, offset, r.stderr.strip()[:300])
    return frames


def _last_frame(clip_path, jpg, ffmpeg):
    """The clip's last frame as jpg. Live 2026-11-09 event 438a4d8b: the only key moment was the
    event's end, 25.295 s into a 25.282 s clip, so the event got no tags and no summary."""
    try:
        r = subprocess.run([ffmpeg, "-v", "error", "-y", "-sseof", "-1", "-i", clip_path,
                            "-update", "1", "-q:v", "3", jpg],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("ffmpeg failed for the last frame of %s: %r", clip_path, e)
        return False
    return r.returncode == 0 and os.path.exists(jpg)


def _min_clip_age_s():
    """Long enough for a queued summary to still find its frames: the longest a task can wait
    (DIYKEY_TASK_DROP_AGE_S) plus a margin for the tagging before it was queued."""
    try:
        return float(os.environ.get("DIYKEY_TASK_DROP_AGE_S") or 900) + 300
    except ValueError:
        return 1200.0


def prune(clips_dir, keep=KEEP_EVENTS, min_age_s=None):
    """Keep the newest `keep` clips (and their frame folders), and any newer than min_age_s."""
    if min_age_s is None:
        min_age_s = _min_clip_age_s()
    try:
        clips = sorted((e for e in os.scandir(clips_dir) if e.name.endswith(".mp4")),
                       key=lambda e: e.stat().st_mtime, reverse=True)
    except OSError:
        return
    for e in clips[keep:]:
        try:
            if time.time() - e.stat().st_mtime < min_age_s:
                continue
            os.remove(e.path)
            if os.path.exists(e.path + ".start"):
                os.remove(e.path + ".start")
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
        try:
            if time.time() - e.stat().st_mtime < min_age_s:
                continue
            shutil.rmtree(e.path, ignore_errors=True)
        except OSError as err:
            log.warning("could not prune %s: %r", e.path, err)
    # Deep-mode describe and re-ID crops are not clips; prune them by the same age rule.
    for sub in ("describe", "embed", "segment"):
        directory = os.path.join(clips_dir, sub)
        try:
            files = sorted((e for e in os.scandir(directory) if e.is_file()),
                           key=lambda e: e.stat().st_mtime, reverse=True)
        except OSError:
            continue
        for e in files[keep:]:
            try:
                if time.time() - e.stat().st_mtime < min_age_s:
                    continue
                os.remove(e.path)
            except OSError as err:
                log.warning("could not prune %s: %r", e.path, err)


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
