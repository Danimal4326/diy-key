# SPDX-License-Identifier: AGPL-3.0-or-later
"""Second Stage Verification (RequestAI -> second_verifier_mlabel).

Protect sends the export URL for the event plus thumbnailMeta: per thumbnail
time, the tracked objects with ``coord`` = [x, y, w, h] of the box's top-left
corner in 0..1000 of the frame (checked against a live person box). For each
object we say what is really in that box: Protect sets the thumbnail and smart
object to ``detectedAs`` (person / vehicle / animal) with confidence
round(100 * detectionConfidence), keeping the old ones as preReverification*.

The box is matched against the COCO detector (ram_tagger.RamTagger) on the
full frame by IoU; when nothing overlaps, the box (with a margin) is run on its
own, which helps small or IR-washed objects. No match leaves the original type
with a low confidence.
"""
import logging
import time

import cv2

log = logging.getLogger("diykey.verify")

# COCO label -> Protect object type.
VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"}
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
MIN_IOU = 0.3
NO_MATCH_CONFIDENCE = 0.05
CROP_MARGIN = 0.3


def protect_type(label):
    if label == "person":
        return "person"
    if label in VEHICLES:
        return "vehicle"
    if label in ANIMALS:
        return "animal"
    return None


def _rois(entry):
    """roi is one object in Protect's schema but a list on some paths."""
    roi = entry.get("roi")
    if isinstance(roi, dict):
        return [roi]
    return [r for r in roi or () if isinstance(r, dict)]


def box_px(coord, width, height):
    """[x, y, w, h] in 0..1000 -> pixel (x, y, w, h)."""
    x, y, w, h = (float(v) for v in coord[:4])
    return x * width / 1000, y * height / 1000, w * width / 1000, h * height / 1000


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _crop_best(tagger, img, box):
    """Best (type, score) the detector finds inside the box plus a margin, or (None, 0)."""
    h, w = img.shape[:2]
    x, y, bw, bh = box
    mx, my = bw * CROP_MARGIN, bh * CROP_MARGIN
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, 0.0
    best = (None, 0.0)
    for label, score, _ in tagger.detect(img[y0:y1, x0:x1]):
        kind = protect_type(label)
        if kind and score > best[1]:
            best = (kind, score)
    return best


def classify(tagger, img, roi):
    """One verificationResults row's verdict for a roi: (detectedAs, confidence, iou, how)."""
    h, w = img.shape[:2]
    box = box_px(roi.get("coord") or [0, 0, 0, 0], w, h)
    best = (None, 0.0, 0.0)
    for label, score, det in tagger.detect(img):
        kind = protect_type(label)
        overlap = iou(box, det)
        if kind and overlap >= MIN_IOU and score > best[1]:
            best = (kind, score, overlap)
    if best[0]:
        return best[0], best[1], best[2], "frame"
    kind, score = _crop_best(tagger, img, box)
    if kind:
        return kind, score, None, "crop"
    return None, 0.0, None, "none"


def verify(tagger, frames, params):
    """reverificationInferenceSchema result from {thumbnail_ms: jpeg_path} and the task params."""
    t0 = time.monotonic()
    rows = []
    for entry in params.get("thumbnailMeta") or []:
        ts = entry.get("ts")
        img = cv2.imread(frames[ts]) if ts in frames else None
        for roi in _rois(entry):
            claimed = str(roi.get("objectType") or (roi.get("attributes") or {}).get("objectType")
                          or "").lower()
            tracker = roi.get("trackerID", roi.get("trackerId"))
            row = {"thumbnailMs": ts, "trackerID": int(tracker) if tracker is not None else -1,
                   "objectType": claimed, "iouScore": None, "detectionConfidenceList": []}
            if img is None:
                row.update(isValidDetection=True, detectedAs=claimed, modelRunStatus="noFrame",
                           detectionConfidence=(roi.get("confidence") or 0) / 100)
            else:
                kind, score, overlap, how = classify(tagger, img, roi)
                if kind:
                    row.update(isValidDetection=kind == claimed, detectedAs=kind,
                               detectionConfidence=round(score, 3), iouScore=overlap,
                               modelRunStatus=f"success:{how}")
                else:
                    row.update(isValidDetection=False, detectedAs=claimed,
                               detectionConfidence=NO_MATCH_CONFIDENCE, modelRunStatus="noMatch")
            rows.append(row)
            log.info("verify %s @%s tracker %s: claimed %s -> %s %.2f (%s)", params.get("event"),
                     ts, row["trackerID"], claimed, row["detectedAs"], row["detectionConfidence"],
                     row["modelRunStatus"])
    infer_ms = int((time.monotonic() - t0) * 1000)
    return {"cameraId": params.get("camera", ""), "eventId": params.get("event", ""),
            "status": "success", "scoreThresholdApplied": params.get("score_threshold", 0.8),
            "preProcessMs": 0, "inferMs": infer_ms, "timeElapsedMs": infer_ms,
            "verificationResults": rows}
