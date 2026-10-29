# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find Anything over the AI Key's nl-search connection.

A real AI Key opens a second WebSocket, wss://<console>:7443/wss/nl-search/v1
(x-ident, subprotocol ucp4, its pinned client cert). While one is open, Protect
sends every Find Anything query there instead of to a console AI Controller:

  NL_PARSE     {querySentence, model}  -> {keyTags: [{matchedWord, tags}], objectTypes,
                                           startTime?, endTime?, timeTag?, txtEmbed,
                                           model, dim, exact_match}
  IMAGE_SEARCH {imgUri}                -> {imgEmbed}

and then queries the AI Key's ramDetections: tag ids for keyTags, cosine distance
for txtEmbed/imgEmbed. Tags come from Protect's RAM vocabulary (DIYKEY_RAM_TAGS:
path to Protect's app/fixtures/ram_tags.csv; a small built-in list otherwise).
Embeddings are clip_embed's CLIP ViT-L/14, the space DIY-Key stores images in.
"""
import datetime
import logging
import os
import re

import clip_embed
import ram_tagger

log = logging.getLogger("diykey.nlsearch")

PATH = "/wss/nl-search/v1"
PORT = 7443

OBJECT_WORDS = {
    "person": {"person", "people", "man", "men", "woman", "women", "kid", "kids", "child",
               "children", "boy", "girl", "someone", "pedestrian", "human"},
    "vehicle": {"car", "cars", "truck", "trucks", "van", "suv", "vehicle", "vehicles", "bus",
                "motorcycle", "motorbike", "bike", "bicycle", "pickup", "trailer"},
    "animal": {"dog", "dogs", "cat", "cats", "animal", "animals", "deer", "bird", "birds", "fox",
               "raccoon", "bear", "horse", "squirrel"},
}
STOP = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or", "to", "by", "near",
        "show", "me", "find", "any", "some", "is", "are", "was", "who", "that", "wearing"}
_vocab = None


def vocabulary():
    """RAM tag names Protect accepts (lowercase)."""
    global _vocab
    if _vocab is None:
        names = set()
        path = os.path.expanduser(os.environ.get("DIYKEY_RAM_TAGS") or "")
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 3 and parts[2].strip().lower() == "true":
                        names.add(parts[1].strip().lower())
        except OSError:
            pass
        if not names:
            names = {t for t in ram_tagger.COCO_TO_RAM.values() if t}
            names |= {"person", "car", "truck", "bus", "dog", "cat", "bicycle", "motorcycle",
                      "night", "man", "woman", "suv", "van", "trailer", "driveway", "yard"}
            names |= ram_tagger.COLOURS
        log.info("nl-search vocabulary: %d tags%s", len(names), f" from {path}" if path else "")
        _vocab = names
    return _vocab


def _time_range(text, now=None):
    """(start_ms, end_ms, phrase) for a time phrase in text, or (None, None, None)."""
    now = now or datetime.datetime.now()
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    ms = lambda d: int(d.timestamp() * 1000)  # noqa: E731
    m = re.search(r"\b(?:in the )?(?:last|past) (\d+) (minute|hour|day|week)s?\b", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"minute": datetime.timedelta(minutes=n), "hour": datetime.timedelta(hours=n),
                 "day": datetime.timedelta(days=n), "week": datetime.timedelta(weeks=n)}[unit]
        return ms(now - delta), ms(now), m.group(0)
    fixed = [
        (r"\blast night\b", day - datetime.timedelta(hours=6), day + datetime.timedelta(hours=6)),
        (r"\bthis morning\b", day + datetime.timedelta(hours=5), day + datetime.timedelta(hours=12)),
        (r"\btonight\b", day + datetime.timedelta(hours=18), now),
        (r"\byesterday\b", day - datetime.timedelta(days=1), day),
        (r"\btoday\b", day, now),
        (r"\bthis week\b", day - datetime.timedelta(days=day.weekday()), now),
    ]
    for pattern, start, end in fixed:
        m = re.search(pattern, text)
        if m:
            return ms(start), ms(end), m.group(0)
    return None, None, None


def parse(query, now=None):
    """NL_PARSE reply for a query sentence."""
    text = " ".join(query.lower().split())
    vocab = vocabulary()
    start, end, phrase = _time_range(text, now)
    retrieval = " ".join(text.replace(phrase, " ").split()) if phrase else text
    words = [w.strip(".,!?\"'") for w in retrieval.split()]
    key_tags, seen = [], set()
    for size in (3, 2, 1):  # longest vocabulary phrase first ("police car" over "car")
        for i in range(len(words) - size + 1):
            gram = " ".join(words[i:i + size])
            if gram in seen or (size == 1 and gram in STOP) or gram not in vocab:
                continue
            if any(gram in s for s in seen):
                continue
            seen.add(gram)
            key_tags.append({"matchedWord": gram, "tags": [gram]})
    object_types = [t for t, ws in OBJECT_WORDS.items() if any(w in ws for w in words)]
    reply = {"keyTags": key_tags, "objectTypes": object_types,
             "txtEmbed": clip_embed.text_embed(retrieval or text) or [],
             "model": clip_embed.MODEL_NAME, "dim": clip_embed.DIM}
    if start is not None:
        reply.update(startTime=start, endTime=end, timeTag=phrase)
    if not key_tags:
        reply["exact_match"] = start is not None  # a pure time query lists everything in it
    elif len(key_tags) == 1:
        # Protect's exact mode searches by object label and returns nothing without one, so only
        # a bare object word ("car", "person") is exact; "night", "grey car" go to vector/hybrid.
        bare = retrieval.strip()
        reply["exact_match"] = bare in vocab and any(bare in ws for ws in OBJECT_WORDS.values())
    return reply


def image_search(img_bgr):
    """IMAGE_SEARCH reply for the fetched image."""
    return {"imgEmbed": clip_embed.image_embed(img_bgr) or []}
