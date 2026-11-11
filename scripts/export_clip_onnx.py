#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export the CLIP model DIY-Key uses to ONNX, for DIYKEY_CLIP_ONNX (no torch at runtime).

Writes <out>/clip-vit-l14-visual.onnx (~1.2 GB), clip-vit-l14-text.onnx (~0.5 GB) and
tokenizer.json: open_clip's ViT-L-14-quickgelu with OpenAI's weights (the model clip_embed's
torch backend loads), each encoder returning the raw 768-d embedding (clip_embed normalises),
and OpenAI's CLIP tokenizer from Hugging Face (openai/clip-vit-large-patch14), whose tokens match
open_clip's. Then it checks the ONNX files against torch on a few inputs.

Needs, on the machine that runs it (not the AI Key): torch, open_clip_torch, onnx, onnxruntime,
tokenizers, huggingface_hub, Pillow, opencv-python-headless, numpy.

  python3 scripts/export_clip_onnx.py /path/to/models/clip
  # then DIYKEY_CLIP=1 DIYKEY_CLIP_ONNX=/path/to/models/clip (and DIYKEY_ORT_DEVICE=GPU)
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "diykey"), os.path.join(ROOT, "diykey", "stage3_wss")]


def main(out):
    import numpy as np
    import onnxruntime as ort
    import open_clip
    import torch
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    import clip_embed

    os.makedirs(out, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    model.eval()

    class Visual(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixels):
            return self.m.encode_image(pixels)

    class Text(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tokens):
            return self.m.encode_text(tokens)

    tok = Tokenizer.from_file(hf_hub_download("openai/clip-vit-large-patch14", "tokenizer.json"))
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (h, w, 3), np.uint8) for h, w in ((480, 640), (100, 37))]
    phrases = ["a photo of person", "a white car in the driveway at night"]
    pix = clip_embed.preprocess(imgs[0])
    ids = clip_embed.tokens(tok, phrases[0])
    for name, module, example, inp in (("visual", Visual(model), pix, "pixels"),
                                       ("text", Text(model), ids, "tokens")):
        path = os.path.join(out, clip_embed.VISUAL_ONNX if name == "visual" else clip_embed.TEXT_ONNX)
        torch.onnx.export(module.eval(), (torch.from_numpy(example),), path, input_names=[inp],
                          output_names=["embedding"],
                          dynamic_axes={inp: {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=17, dynamo=False)
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
    shutil.copyfile(hf_hub_download("openai/clip-vit-large-patch14", "tokenizer.json"),
                    os.path.join(out, clip_embed.TOKENIZER_JSON))

    reference = open_clip.get_tokenizer("ViT-L-14-quickgelu")
    vis = ort.InferenceSession(os.path.join(out, clip_embed.VISUAL_ONNX), providers=["CPUExecutionProvider"])
    txt = ort.InferenceSession(os.path.join(out, clip_embed.TEXT_ONNX), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        for img in imgs:
            want = model.encode_image(torch.from_numpy(clip_embed.preprocess(img))).numpy()[0]
            got = vis.run(None, {"pixels": clip_embed.preprocess(img)})[0][0]
            cos = float(want @ got / np.linalg.norm(want) / np.linalg.norm(got))
            assert cos > 0.9999, f"image embedding differs from torch (cosine {cos})"
        for phrase in phrases:
            assert np.array_equal(clip_embed.tokens(tok, phrase), reference([phrase]).numpy()), phrase
            want = model.encode_text(reference([phrase])).numpy()[0]
            got = txt.run(None, {"tokens": clip_embed.tokens(tok, phrase)})[0][0]
            cos = float(want @ got / np.linalg.norm(want) / np.linalg.norm(got))
            assert cos > 0.9999, f"text embedding differs from torch (cosine {cos})"
    print(f"checked against torch: tokens identical, embeddings match. Set DIYKEY_CLIP_ONNX={out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
