from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_motion_features(frame_index: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Compute frame-diff energy. *out_csv* should already be fps-keyed by the caller."""
    if out_csv.exists():
        return pd.read_csv(out_csv)

    rows = frame_index.sort_values("t_sec").to_dict("records")
    diffs = []
    prev = None
    for row in tqdm(rows, desc="  motion", unit="fr", leave=False):
        img = cv2.imread(row["frame_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            energy = 0.0
        elif prev is None:
            energy = 0.0
        else:
            diff = cv2.absdiff(prev, img)
            energy = float(np.mean(diff))
        diffs.append({"video_id": row["video_id"], "t_sec": row["t_sec"], "diff_energy": energy})
        prev = img

    df = pd.DataFrame(diffs)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
