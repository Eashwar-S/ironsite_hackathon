from __future__ import annotations

import numpy as np

LABELS = ["IDLE", "WORKING", "TRANSIT"]

DEFAULT_LABEL_PROMPTS = {
    "IDLE": "a person standing idle at a construction site or using his phone or talking to someone",
    "WORKING": "a person working at a construction site with tools or equipment or materials except mobile phones",
    "TRANSIT": "a person strictly walking or moving at a construction site without his hands in the frame",
}


def zero_shot_classify(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    logit_scale: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify frames via cosine similarity between image and text embeddings.

    Args:
        image_embeddings: (N, D) image embeddings from CLIP/SigLIP vision encoder.
        text_embeddings: (C, D) text embeddings, one per class.
        logit_scale: Temperature scaling factor (CLIP default ≈ 100).

    Returns:
        pred: (N,) predicted class indices.
        probs: (N, C) softmax probabilities.
    """
    # L2 normalise
    img_norm = image_embeddings / (
        np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8
    )
    txt_norm = text_embeddings / (
        np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8
    )

    # cosine similarity → logits
    logits = (img_norm @ txt_norm.T) * logit_scale

    # numerically-stable softmax
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    pred = np.argmax(probs, axis=1)
    return pred, probs
