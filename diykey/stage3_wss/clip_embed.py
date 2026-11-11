# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLIP ViT-L/14 embeddings for Protect's search (768-d, unit length).

Protect searches the AI Key's ramDetections by cosine distance between a query's
text embedding (NL_PARSE txtEmbed, model "clip-ViT-L-14") and each stored
object's image embedding (thumbnailTags imgEmbed). DIY-Key makes both, so they
share one space: open_clip's ViT-L-14 with OpenAI's weights (the quickgelu variant
those weights were trained with), the public model
Protect names. Like the AI Key's text encoder, short queries get "a photo of ".

Off unless DIYKEY_CLIP=1. Loaded once, used from several threads under a lock. Two backends,
same vectors:

  torch (default)  open_clip_torch + torch; ~1.7 GB of weights downloaded on first use. Runs on
                   Apple's GPU (MPS), CUDA or the CPU; FP16 below.
  ONNX             DIYKEY_CLIP_ONNX=<dir> holding clip-vit-l14-visual.onnx,
                   clip-vit-l14-text.onnx and tokenizer.json (scripts/export_clip_onnx.py makes
                   them from the same open_clip model). onnxruntime + Pillow + tokenizers, no
                   torch; runs where ort_device.py says (DIYKEY_ORT_DEVICE=GPU: an Intel iGPU).

Checked 2026-11-10 against the torch backend on real camera frames and crops: preprocessing
identical (max difference 0 over 29 images), tokens identical (14 704 texts: every Protect tag
plain and as "a photo of ...", queries, descriptions), embeddings cosine 1.00000 on CPU and
>= 0.99962 on an Iris Xe (FP16). Per image torch CPU 640-690 ms, onnxruntime CPU 369 ms,
OpenVINO iGPU 161 ms; per text 61 / 44 / 18 ms.

DIYKEY_CLIP_FP16=1 (torch backend; opt-in, MPS/CUDA only; CPU stays FP32) runs CLIP in FP16 from an FP16 copy
of the same weights (DIYKEY_CLIP_FP16_WEIGHTS, default ~/.diy-key/models/<FP16_FILE>; written
once from the FP32 checkpoint when missing, restart afterwards to drop that copy's memory). The
model is built on the meta device and the FP16 tensors land straight on the GPU, so no FP32 copy
stays in the process: plain .half() after an FP32 load saved nothing (+2.4 GB either way, the
allocator keeps the FP32 pages). Mac B (M1 Pro) 2026-11-10, after 16 frames: +1,088 MB vs +2,832 MB
FP32 (Mac A logged +3,077 MB); 44 Driveway crops x 15 queries moved cosine by <= 0.0009, i.e. <= 0.28
points of Protect's (cos-0.08)/0.32 similarity, no result crossed 50 %, every top-3 the same.
FP32-stored image vectors still match FP16 queries (<= 0.04 points), so no re-index is needed.
docs/memory-speed-2026-11.md.
"""
import logging
import os
import threading

import cv2
import numpy as np

log = logging.getLogger("diykey.clip")

MODEL_NAME = "clip-ViT-L-14"
DIM = 768
ONNX_ENV = "DIYKEY_CLIP_ONNX"
VISUAL_ONNX, TEXT_ONNX, TOKENIZER_JSON = (
    "clip-vit-l14-visual.onnx", "clip-vit-l14-text.onnx", "tokenizer.json")
# open_clip's ViT-L-14-quickgelu / openai preprocessing and tokenizer framing.
SIZE = 224
MEAN = np.array((0.48145466, 0.4578275, 0.40821073), np.float32)
STD = np.array((0.26862954, 0.26130258, 0.27577711), np.float32)
SOT, EOT, CONTEXT = 49406, 49407, 77
_lock = threading.Lock()
_model = None
_model_key = None


OPEN_CLIP_NAME = "ViT-L-14-quickgelu"
FP16_FILE = "clip-ViT-L-14-quickgelu-openai-fp16.safetensors"


def enabled():
    return os.environ.get("DIYKEY_CLIP", "0").strip().lower() in ("1", "true", "yes", "on")


def fp16_wanted(device):
    """DIYKEY_CLIP_FP16=1 on a GPU device. CPU FP16 is slow and partly unsupported, so a CPU
    host keeps FP32 whatever the setting."""
    return os.environ.get("DIYKEY_CLIP_FP16", "0") == "1" and device in ("mps", "cuda")


def fp16_weights_path():
    return os.path.expanduser(os.environ.get("DIYKEY_CLIP_FP16_WEIGHTS")
                              or os.path.join("~/.diy-key/models", FP16_FILE))


def _write_fp16_weights(open_clip, path):
    """One-time: the OpenAI FP32 checkpoint as FP16 safetensors (atomic rename)."""
    from safetensors.torch import save_file
    model, _, _ = open_clip.create_model_and_transforms(OPEN_CLIP_NAME, pretrained="openai",
                                                        device="cpu")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    save_file({k: v.half().contiguous() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, path)
    log.info("CLIP: wrote FP16 weights %s; restart to free the FP32 load's memory", path)


def _load_fp16(open_clip, torch, device):
    """(model, preprocess) in FP16 with no FP32 copy left on the device."""
    from safetensors.torch import load_file
    path = fp16_weights_path()
    if not os.path.isfile(path):
        _write_fp16_weights(open_clip, path)
    with torch.device("meta"):
        model = open_clip.create_model(OPEN_CLIP_NAME, pretrained=None, device="meta")
    model = model.to_empty(device=device)
    model.load_state_dict(load_file(path, device=device), assign=True)
    # attn_mask is a non-persistent buffer (not in the weights): rebuild the causal mask.
    n = model.attn_mask.shape[0]
    model.attn_mask = torch.full((n, n), float("-inf"), device=device).triu_(1)
    if device == "mps":
        # load_file(device="mps") leaves ~2 GB of staging buffers in the MPS driver; the first
        # image made them resident (+845 -> +2,085 MB on Mac B) unless they are released here.
        torch.mps.empty_cache()
    cfg = open_clip.get_pretrained_cfg(OPEN_CLIP_NAME, "openai")
    preprocess = open_clip.image_transform(
        model.visual.image_size, is_train=False, mean=cfg["mean"], std=cfg["std"],
        interpolation=cfg.get("interpolation", "bicubic"),
        resize_mode=cfg.get("resize_mode", "shortest"))
    return model, preprocess


def _load_key():
    path = fp16_weights_path()
    try:
        weights = os.path.getmtime(path) if os.path.exists(path) else None
    except OSError:
        weights = None
    return (OPEN_CLIP_NAME, os.environ.get("DIYKEY_CLIP_FP16", "0"), weights,
            (os.environ.get(ONNX_ENV) or "").strip())


class _TorchBackend:
    def __init__(self):
        import open_clip
        import torch
        self.torch = torch
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        fp16 = fp16_wanted(self.device)
        if fp16:
            self.model, self.preprocess = _load_fp16(open_clip, torch, self.device)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                OPEN_CLIP_NAME, pretrained="openai", device=self.device)
        self.model.eval()
        self.dtype = next(self.model.parameters()).dtype
        self.tokenizer = open_clip.get_tokenizer(OPEN_CLIP_NAME)
        self.where = f"torch {self.device} ({'fp16' if fp16 else 'fp32'})"

    def image(self, img_bgr):
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with self.torch.no_grad():
            x = self.preprocess(pil).unsqueeze(0).to(self.device, self.dtype)
            feats = self.model.encode_image(x)
        return feats.float().cpu().numpy()

    def text(self, phrase):
        with self.torch.no_grad():
            feats = self.model.encode_text(self.tokenizer([phrase]).to(self.device))
        return feats.float().cpu().numpy()


def preprocess(img_bgr):
    """[1, 3, 224, 224] float32, exactly open_clip's transform: shortest side to 224 (bicubic,
    antialiased, as torchvision's Resize on a PIL image), centre crop, CLIP mean/std."""
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    w, h = pil.size
    short, long_ = (w, h) if w <= h else (h, w)
    other = int(SIZE * long_ / short)
    nw, nh = (SIZE, other) if w <= h else (other, SIZE)
    pil = pil.resize((nw, nh), Image.BICUBIC)
    top, left = int(round((nh - SIZE) / 2.0)), int(round((nw - SIZE) / 2.0))
    pil = pil.crop((left, top, left + SIZE, top + SIZE))
    x = (np.asarray(pil, np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def tokens(tokenizer, phrase):
    """[1, 77] int64 as open_clip frames them: start, BPE ids (truncated), end, zero padding."""
    ids = tokenizer.encode(phrase, add_special_tokens=False).ids[:CONTEXT - 2]
    row = np.zeros((1, CONTEXT), np.int64)
    row[0, :len(ids) + 2] = [SOT] + ids + [EOT]
    return row


class _OnnxBackend:
    def __init__(self, folder):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        import ort_device
        cpu = ["CPUExecutionProvider"]
        self.visual = ort_device.session(ort, os.path.join(folder, VISUAL_ONNX), cpu, "CLIP image")
        self.textual = ort_device.session(ort, os.path.join(folder, TEXT_ONNX), cpu, "CLIP text")
        self.tokenizer = Tokenizer.from_file(os.path.join(folder, TOKENIZER_JSON))
        self.v_in = self.visual.get_inputs()[0].name
        self.t_in = self.textual.get_inputs()[0].name
        self.where = (f"ONNX, image on {ort_device.describe(self.visual)}, "
                      f"text on {ort_device.describe(self.textual)}")

    def image(self, img_bgr):
        return self.visual.run(None, {self.v_in: preprocess(img_bgr)})[0]

    def text(self, phrase):
        return self.textual.run(None, {self.t_in: tokens(self.tokenizer, phrase)})[0]


def _load():
    """The loaded backend, or None when CLIP is unavailable (logged once)."""
    global _model, _model_key
    key = _load_key()
    with _lock:
        if _model is False and _model_key != key:
            # The FP16 weights appeared or changed, the toggle or the ONNX folder moved: retry once.
            _model = None
        if _model is None:
            _model_key = key
            folder = key[-1]
            try:
                _model = _OnnxBackend(os.path.expanduser(folder)) if folder else _TorchBackend()
                log.info("CLIP ViT-L-14 (openai) loaded: %s", _model.where)
            except Exception as e:  # noqa: BLE001 -- search just stays without embeddings
                log.warning("CLIP unavailable%s: %r", f" ({ONNX_ENV}={folder})" if folder else "", e)
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
    with _lock:
        feats = loaded.image(img_bgr)
    return _unit(feats)


def text_embed(text):
    """768 floats for a search phrase, prefixed like the AI Key's encoder; or None."""
    if not enabled():
        return None
    loaded = _load()
    if not loaded:
        return None
    phrase = text.strip()
    if len(phrase.split()) < 3:
        phrase = "a photo of " + phrase
    with _lock:
        feats = loaded.text(phrase)
    return _unit(feats)


# Zero-shot labelling (ram_tagger.not_animal_prob): each label is the mean of these prompts, as
# in CLIP's own zero-shot recipe; the second covers IR night crops, which most of ours are. Both
# have 3+ words, so text_embed adds no prefix. Text vectors are cached per label set.
LABEL_TEMPLATES = ("a photo of a {}.", "a black and white night photo of a {}.")
_label_cache = {}


def label_matrix(labels):
    """Unit text vectors, one row per label (cached), or None without CLIP."""
    key = tuple(labels)
    if key not in _label_cache:
        rows = []
        for label in labels:
            vecs = [text_embed(t.format(label)) for t in LABEL_TEMPLATES]
            if any(v is None for v in vecs):
                return None
            v = np.mean(np.asarray(vecs, np.float32), 0)
            rows.append(v / np.linalg.norm(v))
        _label_cache[key] = np.stack(rows)
    return _label_cache[key]


def label_probs(image_vec, labels):
    """{label: probability} for a unit image vector over these labels (softmax of 100 x cosine,
    CLIP's logit scale), or None without CLIP or an image vector."""
    if not image_vec:
        return None
    m = label_matrix(labels)
    if m is None:
        return None
    logits = 100.0 * (m @ np.asarray(image_vec, np.float32))
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return dict(zip(labels, (float(x) for x in p)))


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
