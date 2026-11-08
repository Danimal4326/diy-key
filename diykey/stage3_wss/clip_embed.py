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


REID_DIM = 1024
REID_MODEL = "diykey-clip-vitl14-pool"


def reid_embed(img_bgr):
    """1024 floats, unit length, for a person crop (BGR), or None. Protect's generate-embeddings
    reply wants a 1024-d `reidEmbed` per person (M/315148:83 length(1024); M/761379:380-400;
    stored as ramDetections.reidEmbedding and grouped by M/186873 matchSession). It only ever
    compares these with each other (cosine), so any consistent 1024-d space works, though this is
    NOT the firmware's OSNet-style model: it is ViT-L/14's pooled token BEFORE the final
    projection (visual.ln_post output, 1024 wide; the 768-d image_embed is that times `proj`).
    reidCutoffDefault (0.75, M/186873:48-51) was calibrated for the firmware model; Protect does
    not send the cutoff in the task, so it cannot be logged here, and CLIP similarities between
    different people on one camera may sit above it (strangers merged) -- tune
    deepUnderstandingSettings.reidCutoffDefault/ByCamera if so. Reuses the loaded CLIP model.

    Firmware reference (aikey-fw 2.2.8 ui-reid-agent agent.py + trt_engine.py): the real agent
    has no generate-embeddings route (it serves /segment, /segment-video, /search-reranking, which
    reranks with cosine distance 1 - dot, so its vectors are unit); its encoder resizes a person
    crop to 128x384 (OSNet) and returns the engine output unnormalised. We keep CLIP's own
    224 preprocessing because the ViT was trained on it."""
    if not enabled() or img_bgr is None or img_bgr.size == 0:
        return None
    loaded = _load()
    if not loaded:
        return None
    model, preprocess, _, device = loaded
    import torch
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    grabbed = []
    hook = model.visual.ln_post.register_forward_hook(lambda _m, _i, out: grabbed.append(out))
    try:
        with _lock, torch.no_grad():
            model.encode_image(preprocess(pil).unsqueeze(0).to(device))
    finally:
        hook.remove()
    if not grabbed:
        return None
    pooled = grabbed[-1].float().cpu().numpy()
    if pooled.ndim == 3:  # older open_clip applies ln_post to every token; CLS is first
        pooled = pooled[:, 0]
    vec = pooled.reshape(-1)
    return _unit(vec) if vec.size == REID_DIM else None


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


def protect_similarity(object_vec, query_vec):
    """Protect's similarity in percent, as AI Key Advanced alarms score it: pgvector
    `(1 - (a <=> b) - 0.08) / 0.32`, `<=>` being cosine distance, and the rule fires when
    100 * that >= its threshold (M/718745_aiprocessors.js:42-48). Search maps the same way."""
    a = np.asarray(object_vec, dtype=np.float64)
    b = np.asarray(query_vec, dtype=np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return 100 * (cos - 0.08) / 0.32


def blend_weight():
    """DIYKEY_CLIP_TAG_BLEND, default 0.1; an unparseable value means the default."""
    try:
        return float(os.environ.get("DIYKEY_CLIP_TAG_BLEND", "0.1"))
    except ValueError:
        log.warning("DIYKEY_CLIP_TAG_BLEND=%r is not a number; using 0.1",
                    os.environ.get("DIYKEY_CLIP_TAG_BLEND"))
        return 0.1


def blend(image_vec, tags, weight=None):
    """The stored object embedding: the image embedding nudged toward the text of its own tags
    ("a photo of car, night"), unit length. Stock CLIP puts small, blurred, IR-washed camera crops
    only ~0.2 cosine from a matching query; Protect's similarity maps cosine 0.08..0.40 to
    0..100 % and the UI asks for 50 %, so unblended crops never show. Weight 0.1
    (DIYKEY_CLIP_TAG_BLEND) lifted "night"/"car" on a night car crop to 51-68 % while
    "person"/"dog"/"bicycle" stayed at 37-41 %.

    This one vector is both the search row and the alarm input: ram_tagger.thumbnail_tags sends
    it as thumbnailTags[].imgEmbed, Protect keeps it as the object's embedding and passes that
    same vector to aiprocessor.matchRules (M/12387_aiprocessors.js:213-231), which scores it
    with protect_similarity against the rule's stored txtEmbed. So the blend moves alarm scores
    exactly as it moves search ranking: queries naming the crop's own tags gain ~15-30 points
    (an alarm at the UI's default threshold can fire at all), while queries the tags don't name
    stay put -- but a wrong tag (detector calls a van "car") now pulls the "car" score up too.
    Higher weights trade alarm precision for recall; 0 gives the raw CLIP calibration, under
    which a crop rarely clears ~40 %. No evidence yet for a better default than 0.1."""
    if not image_vec or not tags:
        return image_vec
    weight = blend_weight() if weight is None else float(weight)
    if weight <= 0:
        return image_vec
    text_vec = text_embed("a photo of " + ", ".join(tags))
    if not text_vec:
        return image_vec
    return _unit(np.asarray(image_vec, dtype=np.float32) + weight * np.asarray(text_vec, dtype=np.float32))
