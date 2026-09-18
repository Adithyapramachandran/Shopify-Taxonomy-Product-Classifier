"""
Optional image-based classification signal.

This module is intentionally isolated so the rest of the pipeline never has
to know or care whether image classification is actually available in a given
deployment:

- In an environment with outbound network access to a model hub (e.g.
  Hugging Face) and a CLIP-family model installed, `_load_model()` loads a
  zero-shot image classifier once per process and `classify_images_against_categories`
  scores each image against the candidate category names.
- In an offline/locked-down environment (like this prototype's sandbox,
  which only allows a short list of package-index domains and has no route to
  a model hub), model loading fails once, is cached as "unavailable", and every
  call after that returns None immediately and cheaply -- so the pipeline
  falls back to text-only classification (requirement #4) without retrying
  a slow failure on every product.

To enable real image classification in production:
    pip install open_clip_torch torch pillow
and ensure the deployment network allows downloading the model weights once
(e.g. from Hugging Face) or bundle the weights into the image.
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import Optional

_MODEL_STATE = {"checked": False, "available": False, "model": None, "preprocess": None, "tokenizer": None}


def _try_load_model():
    if _MODEL_STATE["checked"]:
        return _MODEL_STATE["available"]
    _MODEL_STATE["checked"] = True
    try:
        import torch  # noqa
        import open_clip  # noqa

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        _MODEL_STATE.update(available=True, model=model, preprocess=preprocess, tokenizer=tokenizer)
    except Exception:
        _MODEL_STATE.update(available=False)
    return _MODEL_STATE["available"]


def is_available() -> bool:
    return _try_load_model()


def classify_images_against_categories(image_bytes_list: list[bytes], candidate_labels: Optional[list[str]]) -> Optional[dict]:
    """Zero-shot match a product's image(s) against a short list of candidate
    category full-names (already narrowed down by the text classifier).
    Returns {"best_label": str, "confidence": float} or None if unavailable.
    """
    if not image_bytes_list or not candidate_labels:
        return None
    if not _try_load_model():
        return None

    import torch
    from PIL import Image

    model = _MODEL_STATE["model"]
    preprocess = _MODEL_STATE["preprocess"]
    tokenizer = _MODEL_STATE["tokenizer"]

    text_tokens = tokenizer([f"a product photo of {c.split('>')[-1].strip()}" for c in candidate_labels])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        best_overall = None
        for raw in image_bytes_list:
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            image_input = preprocess(img).unsqueeze(0)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).softmax(dim=-1)[0]
            top_idx = int(sims.argmax())
            score = float(sims[top_idx])
            if best_overall is None or score > best_overall[1]:
                best_overall = (candidate_labels[top_idx], score)

    if best_overall is None:
        return None
    return {"best_label": best_overall[0], "confidence": best_overall[1]}
