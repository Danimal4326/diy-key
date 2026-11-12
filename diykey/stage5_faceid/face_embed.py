# SPDX-License-Identifier: AGPL-3.0-or-later
"""Face embedding (recognition) on the RK3588 NPU via rknnlite.

Model: w600k_mbf (MobileFaceNet, insightface's buffalo_sc pack), fp16
(unquantized, to preserve cosine-similarity precision), 112x112 input,
512-d L2-normalizable embedding output. See ../../protocol.md and
../../todo.md for pipeline status.
"""
import os

import cv2
import numpy as np
from rknnlite.api import RKNNLite

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "w600k_mbf.rknn")

# insightface's canonical 5-point face template for 112x112 alignment
# (left eye, right eye, nose, left mouth corner, right mouth corner).
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image, landmarks, image_size=112):
    """image: BGR numpy array. landmarks: 5x2 array (from FaceDetector.detect).
    Returns a 112x112 BGR aligned crop."""
    dst = ARCFACE_DST.copy()
    if image_size != 112:
        dst *= image_size / 112.0
    transform, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), dst, method=cv2.LMEDS)
    return cv2.warpAffine(image, transform, (image_size, image_size), borderValue=0.0)


class FaceEmbedder:
    """Wraps the MobileFaceNet RKNN model. One instance per process; not
    thread-safe for concurrent inference calls."""

    def __init__(self, model_path=MODEL_PATH):
        self._rknn = RKNNLite()
        if self._rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"failed to load RKNN model: {model_path}")
        if self._rknn.init_runtime() != 0:
            raise RuntimeError("failed to init RKNN runtime")

    def embed(self, aligned_face_bgr):
        """aligned_face_bgr: 112x112 BGR image, e.g. from align_face().
        Returns a raw (non-normalized) 512-d float32 embedding."""
        rgb = aligned_face_bgr[..., ::-1]
        infer_img = np.expand_dims(rgb, axis=0)
        outputs = self._rknn.inference(inputs=[infer_img])
        if outputs is None:
            raise RuntimeError("RKNN inference failed")
        return outputs[0].reshape(-1).astype(np.float32)

    def close(self):
        self._rknn.release()


def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


if __name__ == "__main__":
    import sys
    from face_detect import FaceDetector

    test_img = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "models", "test.jpg")

    detector = FaceDetector()
    embedder = FaceEmbedder()

    img = cv2.imread(test_img)
    faces = detector.detect(img)
    print(f"{len(faces)} face(s) detected")

    embeddings = []
    for x1, y1, x2, y2, score, landmarks in faces:
        aligned = align_face(img, landmarks)
        emb = embedder.embed(aligned)
        embeddings.append(emb)
        print(f"face score={score:.4f} embedding norm={np.linalg.norm(emb):.4f} "
              f"first5={emb[:5]}")

    # sanity check: re-embedding the same aligned crop should be ~identical
    if embeddings:
        x1, y1, x2, y2, score, landmarks = faces[0]
        aligned = align_face(img, landmarks)
        emb2 = embedder.embed(aligned)
        print(f"self-similarity (same face, re-run) = "
              f"{cosine_similarity(embeddings[0], emb2):.6f} (expect ~1.0)")

    detector.close()
    embedder.close()
