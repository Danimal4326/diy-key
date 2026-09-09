"""Combined face detection + embedding pipeline, real inference on the
RK3588 NPU. See face_detect.py/face_embed.py for the individual models.

This is deliberately NOT wired into stage3_wss/ucp4_client.py or
stage4_modelserver/model_server.py yet: the real wire mechanism for
delivering face results to the console is unconfirmed (see
../../protocol.md §7 and ../../todo.md) -- wiring this in now would mean
guessing a dispatch shape and baking it in as if confirmed. Once a live
capture confirms the real RequestAI/recognizeKeyFrames model_name and
result shape for face recognition, call FacePipeline.process() from
that handler instead of duplicating this logic.
"""
import os

import numpy as np

from face_detect import FaceDetector
from face_embed import FaceEmbedder, align_face


class FacePipeline:
    """One NPU-bound instance per process (each of FaceDetector/FaceEmbedder
    binds an rknnlite runtime context on init); not thread-safe."""

    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()

    def process(self, image_bgr, score_thresh=0.5):
        """Returns a list of dicts: {bbox: (x1,y1,x2,y2), score, embedding
        (512-d float32, NOT L2-normalized -- normalize before comparing)}."""
        results = []
        for x1, y1, x2, y2, score, landmarks in self.detector.detect(image_bgr, score_thresh):
            aligned = align_face(image_bgr, landmarks)
            embedding = self.embedder.embed(aligned)
            results.append({
                "bbox": (x1, y1, x2, y2),
                "score": score,
                "embedding": embedding,
            })
        return results

    def close(self):
        self.detector.close()
        self.embedder.close()


if __name__ == "__main__":
    import sys
    import cv2

    test_img = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "models", "test.jpg")

    pipeline = FacePipeline()
    img = cv2.imread(test_img)
    for face in pipeline.process(img):
        x1, y1, x2, y2 = face["bbox"]
        print(f"face @ ({x1} {y1} {x2} {y2}) score={face['score']:.4f} "
              f"embedding_norm={np.linalg.norm(face['embedding']):.4f}")
    pipeline.close()
