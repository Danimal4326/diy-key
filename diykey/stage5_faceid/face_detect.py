# SPDX-License-Identifier: AGPL-3.0-or-later
"""RetinaFace face detection on the RK3588 NPU via rknnlite.

Model: RetinaFace_mobile320 (biubug6/Pytorch_Retinaface, converted through
airockchip/rknn_model_zoo's pipeline), int8-quantized, 320x320 static input.
Confirmed running on real RK3588 NPU hardware (RKNPU v2, driver 0.9.6),
~9ms/frame. See ../../protocol.md and ../../todo.md for pipeline status.
"""
import os
from math import ceil
from itertools import product as product

import cv2
import numpy as np
from rknnlite.api import RKNNLite

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "RetinaFace_mobile320.rknn")
INPUT_SIZE = (320, 320)  # (height, width)
MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
STEPS = [8, 16, 32]
VARIANCES = (0.1, 0.2)


def _letterbox_resize(image, size, bg_color=114):
    target_width, target_height = size
    image_height, image_width, _ = image.shape
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    canvas[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
    return canvas, aspect_ratio, offset_x, offset_y


def _prior_box(image_size):
    anchors = []
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in STEPS]
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in MIN_SIZES[k]:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                cx = (j + 0.5) * STEPS[k] / image_size[1]
                cy = (i + 0.5) * STEPS[k] / image_size[0]
                anchors += [cx, cy, s_kx, s_ky]
    return np.array(anchors).reshape(-1, 4)


def _decode_boxes(loc, priors):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * VARIANCES[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * VARIANCES[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _nms(dets, thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class FaceDetector:
    """Wraps the RetinaFace RKNN model. One instance per process (rknnlite
    binds to the NPU on init_runtime); not thread-safe for concurrent
    inference calls."""

    def __init__(self, model_path=MODEL_PATH):
        self._rknn = RKNNLite()
        if self._rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"failed to load RKNN model: {model_path}")
        if self._rknn.init_runtime() != 0:
            raise RuntimeError("failed to init RKNN runtime")
        self._priors = _prior_box(INPUT_SIZE)

    def detect(self, image, score_thresh=0.5, nms_thresh=0.5):
        """image: BGR numpy array (as from cv2.imread/cv2.VideoCapture).
        Returns list of (x1, y1, x2, y2, score, landmarks[5][2])."""
        img_height, img_width = image.shape[:2]
        model_height, model_width = INPUT_SIZE
        letterboxed, aspect_ratio, offset_x, offset_y = _letterbox_resize(
            image, (model_width, model_height))
        infer_img = np.expand_dims(letterboxed[..., ::-1], axis=0)  # BGR->RGB, add batch dim

        outputs = self._rknn.inference(inputs=[infer_img])
        if outputs is None:
            raise RuntimeError("RKNN inference failed")
        loc, conf, landmarks = outputs

        boxes = _decode_boxes(loc.squeeze(0), self._priors)
        scale = np.array([model_width, model_height, model_width, model_height])
        boxes = boxes * scale // 1
        boxes[..., 0::2] = np.clip((boxes[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
        boxes[..., 1::2] = np.clip((boxes[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)
        scores = conf.squeeze(0)[:, 1]

        lm = _decode_landmarks(landmarks.squeeze(0), self._priors)
        lm = lm * np.array([model_width, model_height] * 5)
        lm[..., 0::2] = np.clip((lm[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
        lm[..., 1::2] = np.clip((lm[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)

        keep_idx = np.where(scores > score_thresh)[0]
        boxes, scores, lm = boxes[keep_idx], scores[keep_idx], lm[keep_idx]
        order = scores.argsort()[::-1]
        boxes, scores, lm = boxes[order], scores[order], lm[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = _nms(dets, nms_thresh)
        results = []
        for i in keep:
            x1, y1, x2, y2, score = dets[i]
            landmark_pts = lm[i].reshape(5, 2)
            results.append((int(x1), int(y1), int(x2), int(y2), float(score), landmark_pts))
        return results

    def close(self):
        self._rknn.release()


def _decode_landmarks(landmarks, priors):
    return np.concatenate([
        priors[:, :2] + landmarks[:, 2 * i:2 * i + 2] * VARIANCES[0] * priors[:, 2:]
        for i in range(5)
    ], axis=1)


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "models", "test.jpg")
    detector = FaceDetector()
    img = cv2.imread(test_img)
    for x1, y1, x2, y2, score, _ in detector.detect(img):
        print(f"face @ ({x1} {y1} {x2} {y2}) score={score:.6f}")
    detector.close()
