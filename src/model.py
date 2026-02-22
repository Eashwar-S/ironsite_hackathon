from __future__ import annotations

import numpy as np

LABELS = ["IDLE", "WORKING", "TRANSIT", "DOWNTIME"]

# kept for backward compat / simple mode — prefer descriptor_banks for full pipeline
DEFAULT_LABEL_PROMPTS = {
    "IDLE": "first person view of standing still at a construction site with no hand activity",
    "WORKING": "first person view of hands actively using tools at a construction site",
    "TRANSIT": "first person view of walking through a construction site",
    "DOWNTIME": "first person view of resting or on a break at a construction site",
}


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def zero_shot_classify(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    logit_scale: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple zero-shot: cosine similarity → softmax.  Kept as fallback."""
    img_norm = image_embeddings / (
        np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8
    )
    txt_norm = text_embeddings / (
        np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8
    )
    logits = (img_norm @ txt_norm.T) * logit_scale
    probs = _softmax(logits)
    return np.argmax(probs, axis=1), probs


def multi_axis_classify(
    image_embeddings: np.ndarray,
    bank_text_embeddings: list[np.ndarray],
    bank_label_weights: list[np.ndarray],
    bank_multipliers: list[float],
    camera_text_embeddings: np.ndarray,
    camera_confidence: np.ndarray,
    logit_scale: float = 100.0,
    num_labels: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify frames using multi-axis descriptor banks.

    Parameters
    ----------
    image_embeddings : (N, D)
        Image features from CLIP/SigLIP.
    bank_text_embeddings : list of (S_b, D)
        Text features for each label-producing bank (4 banks).
    bank_label_weights : list of (S_b, C)
        Label weight vectors for every descriptor in each bank.
    bank_multipliers : list of float
        Per-bank importance weight.
    camera_text_embeddings : (S_cam, D)
        Text features for the camera-state bank.
    camera_confidence : (S_cam,)
        Confidence multiplier for each camera-state descriptor.
    logit_scale : float
        Cosine-similarity temperature.
    num_labels : int
        Number of parent labels (default 4).

    Returns
    -------
    pred : (N,) int
    probs : (N, C) float
    """
    N = image_embeddings.shape[0]
    img_norm = image_embeddings / (
        np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8
    )

    # Accumulate label evidence across all label-producing banks
    label_evidence = np.zeros((N, num_labels), dtype=np.float64)

    for text_emb, lbl_w, bank_w in zip(
        bank_text_embeddings, bank_label_weights, bank_multipliers
    ):
        # Normalize label-weight columns so each sums to 1.
        # This prevents labels with more descriptors from dominating.
        col_sums = lbl_w.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
        lbl_w_norm = lbl_w / col_sums

        txt_norm = text_emb / (
            np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8
        )
        # (N, S_b) cosine similarities
        sim = (img_norm @ txt_norm.T) * logit_scale
        # softmax within bank → descriptor probabilities
        desc_probs = _softmax(sim)  # (N, S_b)
        # weighted sum: descriptor probs × normalized label weights → (N, C)
        bank_evidence = desc_probs @ lbl_w_norm
        label_evidence += bank_w * bank_evidence

    # Camera-state confidence modifier
    cam_norm = camera_text_embeddings / (
        np.linalg.norm(camera_text_embeddings, axis=1, keepdims=True) + 1e-8
    )
    cam_sim = (img_norm @ cam_norm.T) * logit_scale
    cam_probs = _softmax(cam_sim)  # (N, S_cam)
    # expected confidence = weighted sum of descriptor confidences
    confidence = cam_probs @ camera_confidence  # (N,)
    # scale evidence by confidence (low-quality frames → flatter distribution)
    label_evidence *= confidence[:, None]

    # Final softmax over parent labels
    probs = _softmax(label_evidence).astype(np.float32)
    pred = np.argmax(probs, axis=1)
    return pred, probs
