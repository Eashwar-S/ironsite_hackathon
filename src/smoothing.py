from __future__ import annotations

from collections import Counter
import numpy as np


LABELS = ["IDLE", "WORKING", "TRANSIT", "DOWNTIME"]


def majority_smooth(labels: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1:
        return labels
    radius = window // 2
    out = labels.copy()
    for i in range(len(labels)):
        s = max(0, i - radius)
        e = min(len(labels), i + radius + 1)
        vote = Counter(labels[s:e]).most_common(1)[0][0]
        out[i] = vote
    return out


def to_segments(t_sec: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> list[dict]:
    if len(labels) == 0:
        return []
    segments = []
    start_idx = 0
    for i in range(1, len(labels) + 1):
        boundary = i == len(labels) or labels[i] != labels[start_idx]
        if boundary:
            cls = int(labels[start_idx])
            seg_probs = probs[start_idx:i, cls]
            start_sec = float(t_sec[start_idx])
            if i < len(t_sec):
                end_sec = float(t_sec[i])
            elif len(t_sec) > 1:
                end_sec = float(t_sec[i - 1] + (t_sec[i - 1] - t_sec[i - 2]))
            else:
                end_sec = start_sec + 1.0
            segments.append(
                {
                    "label": LABELS[cls],
                    "start_sec": start_sec,
                    "end_sec": max(end_sec, start_sec),
                    "duration_sec": max(0.0, end_sec - start_sec),
                    "confidence": float(np.mean(seg_probs)) if len(seg_probs) else 0.0,
                }
            )
            start_idx = i
    return segments
