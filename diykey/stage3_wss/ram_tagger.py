# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tag key-moment frames for a recognizeKeyFrames (RAM) result.

Protect stores RAM results by tag *name* from its fixed vocabulary
(app/fixtures/ram_tags.csv on the console, ~7.3k tags), keeps every
key-moment tag on the event and the description as ramDescription. Here the
tags come from a COCO object detector (an Ultralytics-style anchor-free ONNX
export, [1, 84, N] output, e.g. Frigate's yolov9-s-320.onnx), plus "night"
for IR (greyscale) frames. COCO names are already vocabulary tags except the
ones mapped or dropped below.
"""
import ast
import logging
import time

import cv2
import numpy as np

import clip_embed

log = logging.getLogger("diykey.ram")

# COCO name -> Protect RAM tag; None = no supported tag for it.
COCO_TO_RAM = {
    "airplane": "plane", "suitcase": "luggage", "skis": "ski", "toilet": "toilet bowl",
    "tv": "television", "refrigerator": "fridge", "potted plant": "plant",
    "dining table": "table", "cell phone": "phone",
    "fire hydrant": None, "teddy bear": None, "sports ball": None, "hot dog": None,
    "scissors": None, "hair drier": None,
}
NIGHT_TAG = "night"


class RamTagger:
    def __init__(self, model_path, conf=0.35, iou=0.5):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        names = self.session.get_modelmeta().custom_metadata_map.get("names")
        self.labels = ast.literal_eval(names) if names else {}
        self.conf, self.iou = conf, iou
        log.info("RAM tagger: %s (%dpx, %d classes)", model_path, self.size, len(self.labels))

    def _letterbox(self, img):
        h, w = img.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        return canvas, scale, left, top

    def detect(self, img_bgr):
        """[(label, score, (x, y, w, h) in image pixels)], after NMS."""
        canvas, scale, left, top = self._letterbox(img_bgr)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        out = self.session.run(None, {self.input_name: x})[0][0].T  # [N, 4 + classes]
        scores = out[:, 4:]
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        keep = best >= self.conf
        boxes, best, cls = out[keep, :4], best[keep], cls[keep]
        rects = [[float((cx - bw / 2 - left) / scale), float((cy - bh / 2 - top) / scale),
                  float(bw / scale), float(bh / scale)] for cx, cy, bw, bh in boxes]
        idx = cv2.dnn.NMSBoxes(rects, best.tolist(), self.conf, self.iou) if rects else []
        return [(self.labels.get(int(cls[i]), str(int(cls[i]))), float(best[i]), tuple(rects[i]))
                for i in np.array(idx).flatten()]

    def tag_frame(self, img_bgr):
        """[{tag, confScore}] for one frame, best score per tag."""
        found = {}
        for label, score, _ in self.detect(img_bgr):
            tag = COCO_TO_RAM.get(label, label)
            if tag and score > found.get(tag, 0):
                found[tag] = score
        if is_greyscale(img_bgr):
            found[NIGHT_TAG] = 0.9
        return [{"tag": t, "confScore": round(s, 3)}
                for t, s in sorted(found.items(), key=lambda kv: -kv[1])]


# Detector labels that are consistent with each Protect object type (crops are of one object).
FAMILY = {
    "person": {"person"},
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"},
    "animal": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
               "giraffe"},
}


def tag_crop(tagger, img_bgr, hint):
    """Tags for a crop of one tracked object. hint is the task's image entry: objectType plus the
    camera's attributes (vehicleType, color). Detector labels outside the object's family are
    dropped (car crops read as boat/train), and the camera's own attributes are added."""
    object_type = str(hint.get("objectType") or (hint.get("attributes") or {}).get("objectType")
                      or "").lower()
    family = FAMILY.get(object_type)
    attrs = hint.get("attributes") or {}
    family_tags = {COCO_TO_RAM.get(lbl, lbl) for lbl in family or ()}
    found = {}
    for t in tagger.tag_frame(img_bgr):
        if t["tag"] == NIGHT_TAG or family is None:
            found[t["tag"]] = t["confScore"]
        elif t["tag"] in family_tags and not (attrs.get("vehicleType") or {}).get("val") \
                and not any(k in family_tags for k in found):
            found[t["tag"]] = t["confScore"]  # one object: only the best label of its family
    for key in ("vehicleType", "color"):
        val = (attrs.get(key) or {}).get("val")
        if val:
            conf = round(min(1.0, (attrs[key].get("confidence") or 0) / 100), 3)
            found[str(val)] = max(found.get(str(val), 0), conf)
    if object_type and object_type not in ("vehicle",) and object_type not in found:
        found[object_type] = round((hint.get("confidence") or 0) / 100, 3)
    return [{"tag": t, "confScore": c} for t, c in sorted(found.items(), key=lambda kv: -kv[1])]


def is_greyscale(img_bgr, tol=3.0):
    """IR night frames come through with (nearly) equal channels."""
    small = cv2.resize(img_bgr, (64, 36)).astype(np.int16)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    return float(np.mean(np.abs(b - g)) + np.mean(np.abs(g - r))) / 2 < tol


COLOURS = {"white", "black", "grey", "gray", "silver", "red", "blue", "green", "yellow",
           "orange", "brown", "purple", "pink", "beige", "gold"}


def describe(tags):
    """One sentence from the tags, e.g. "Person and car detected at night." or "White car
    detected." (a colour tag goes in front of the first thing it can describe)."""
    names = [t["tag"] for t in tags if t["tag"] != NIGHT_TAG]
    colours = [n for n in names if n in COLOURS]
    things = [n for n in names if n not in COLOURS]
    if colours and things:
        things[0] = f"{colours[0]} {things[0]}"
    when = " at night" if any(t["tag"] == NIGHT_TAG for t in tags) else ""
    if not things:
        return f"Motion detected{when}."
    listed = things[0] if len(things) == 1 else ", ".join(things[:-1]) + " and " + things[-1]
    return f"{listed[0].upper()}{listed[1:]} detected{when}."


def crop_box(img, coord, margin=0.1):
    """The box [x, y, w, h] (0..1000 of the frame, top-left) cut out of img, or None."""
    h, w = img.shape[:2]
    x, y, bw, bh = (float(v) for v in coord[:4])
    x, y, bw, bh = x * w / 1000, y * h / 1000, bw * w / 1000, bh * h / 1000
    x0, y0 = max(0, int(x - bw * margin)), max(0, int(y - bh * margin))
    x1, y1 = min(w, int(x + bw * (1 + margin))), min(h, int(y + bh * (1 + margin)))
    return img[y0:y1, x0:x1] if x1 - x0 >= 8 and y1 - y0 >= 8 else None


def thumbnail_tags(tagger, objects):
    """thumbnailTags for tracked objects: [{trackerID, keyMomentMs, img, hint}] ->
    [{trackerID, keyMomentMs, tags, imgEmbed?}]. Protect stores an object's row in the AI
    Key's ramDetections (what search queries) only when it carries imgEmbed, matched to the
    smart-detect object by trackerId and keyMomentMs == detectedAt."""
    out = []
    for o in objects:
        if o.get("img") is None:
            continue
        entry = {"trackerID": int(o["trackerID"]), "keyMomentMs": int(o["keyMomentMs"]),
                 "tags": tag_crop(tagger, o["img"], o.get("hint") or {})}
        emb = clip_embed.blend(clip_embed.image_embed(o["img"]), [t["tag"] for t in entry["tags"]])
        if emb:
            entry["imgEmbed"] = emb
        out.append(entry)
    return out


def build_result(tagger, camera_id, event_id, frames, pre_ms=0, hints=None, objects=None):
    """recognizeKeyFrames result (ramInferenceSchema) from {moment_ms: jpeg_path}; hints maps
    moment_ms -> the image entry when the frame is a crop of one object (image tasks)."""
    t0 = time.monotonic()
    key_moments, merged = [], {}
    for ms in sorted(frames):
        img = cv2.imread(frames[ms])
        if img is None:
            log.warning("could not read frame %s", frames[ms])
            continue
        hint = (hints or {}).get(ms)
        tags = tag_crop(tagger, img, hint) if hint else tagger.tag_frame(img)
        key_moments.append({"keyMomentMs": int(ms), "tags": tags})
        for t in tags:
            merged[t["tag"]] = max(merged.get(t["tag"], 0), t["confScore"])
    thumbs = thumbnail_tags(tagger, objects or [])
    infer_ms = int((time.monotonic() - t0) * 1000)
    all_tags = [{"tag": t, "confScore": s} for t, s in sorted(merged.items(), key=lambda kv: -kv[1])]
    return {
        "cameraId": camera_id,
        "eventId": event_id,
        "status": "success",
        "description": describe(all_tags),
        "keyMomentsTags": key_moments,
        **({"thumbnailTags": thumbs} if thumbs else {}),
        "inferBoxMs": infer_ms,
        "inferTagMs": 0,
        "inferTxtMs": 0,
        "preProcessMs": int(pre_ms),
        "timeElapsedMs": int(pre_ms) + infer_ms,
    }
