# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLIP ViT-L/14 embeddings for Protect's search (768-d, unit length).

Protect searches the AI Key's ramDetections by cosine distance between a query's
text embedding (NL_PARSE txtEmbed, model "clip-ViT-L-14") and each stored
object's image embedding (thumbnailTags imgEmbed). DIY-Key makes both, so they
share one space: open_clip's ViT-L-14 with OpenAI's weights (the quickgelu variant
those weights were trained with), the public model
Protect names. Like the AI Key's text encoder, short queries get "a photo of ".

Off unless DIYKEY_CLIP=1 (torch + open_clip_torch installed; ~1.7 GB of weights
on first use). Loaded once, used from several threads under a lock.
"""
import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger("diykey.clip")

MODEL_NAME = "clip-ViT-L-14"
DIM = 768
_lock = threading.Lock()
_model = None


def enabled():
    return os.environ.get("DIYKEY_CLIP", "0") == "1"


def _load():
    """(model, preprocess, tokenizer, device), or None when unavailable."""
    global _model
    with _lock:
        if _model is None:
            try:
                import open_clip
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-L-14-quickgelu", pretrained="openai", device=device)
                model.eval()
                _model = (model, preprocess, open_clip.get_tokenizer("ViT-L-14-quickgelu"), device)
                log.info("CLIP ViT-L-14 (openai) loaded on %s", device)
            except Exception as e:  # noqa: BLE001 -- search just stays without embeddings
                log.warning("CLIP unavailable: %r", e)
                _model = False
        return _model or None


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else v.tolist()


def image_embed(img_bgr):
    """768 floats for a BGR image (a whole frame or one object's crop), or None."""
    if not enabled() or img_bgr is None or img_bgr.size == 0:
        return None
    loaded = _load()
    if not loaded:
        return None
    model, preprocess, _, device = loaded
    import torch
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    with _lock, torch.no_grad():
        feats = model.encode_image(preprocess(pil).unsqueeze(0).to(device))
    return _unit(feats.float().cpu().numpy())


def text_embed(text):
    """768 floats for a search phrase, prefixed like the AI Key's encoder; or None."""
    if not enabled():
        return None
    loaded = _load()
    if not loaded:
        return None
    model, _, tokenizer, device = loaded
    import torch
    phrase = text.strip()
    if len(phrase.split()) < 3:
        phrase = "a photo of " + phrase
    with _lock, torch.no_grad():
        feats = model.encode_text(tokenizer([phrase]).to(device))
    return _unit(feats.float().cpu().numpy())


def blend(image_vec, tags, weight=None):
    """The stored object embedding: the image embedding nudged toward the text of its own tags
    ("a photo of car, night"), unit length. Stock CLIP puts small, blurred, IR-washed camera crops
    only ~0.2 cosine from a matching query; Protect's similarity maps cosine 0.08..0.40 to
    0..100 % and the UI asks for 50 %, so unblended crops never show. Weight 0.1
    (DIYKEY_CLIP_TAG_BLEND) lifted "night"/"car" on a night car crop to 51-68 % while
    "person"/"dog"/"bicycle" stayed at 37-41 %."""
    if not image_vec or not tags:
        return image_vec
    weight = float(os.environ.get("DIYKEY_CLIP_TAG_BLEND", "0.1") if weight is None else weight)
    if weight <= 0:
        return image_vec
    text_vec = text_embed("a photo of " + ", ".join(tags))
    if not text_vec:
        return image_vec
    return _unit(np.asarray(image_vec, dtype=np.float32) + weight * np.asarray(text_vec, dtype=np.float32))
