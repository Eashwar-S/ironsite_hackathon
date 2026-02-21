from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def extract_video_frames(
    video_id: str,
    video_path: Path,
    out_dir: Path,
    fps: float,
) -> tuple[pd.DataFrame, float]:
    """Extract frames at *fps* and return (frame_index_df, source_fps).

    The frame index CSV is keyed by fps so caches from different extraction
    rates never collide.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # fps-keyed cache so different sampling rates don't share cache files
    index_path = out_dir / f"frame_index_{fps:.4f}.csv"
    if index_path.exists():
        cap.release()
        return pd.read_csv(index_path), source_fps

    sample_every = max(int(round(source_fps / fps)), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    expected_samples = max(1, total_frames // sample_every)

    rows = []
    frame_idx = 0
    saved_idx = 0
    with tqdm(total=expected_samples, desc=f"  frames [{video_id}]", unit="fr", leave=False) as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                t_sec = frame_idx / source_fps
                frame_path = out_dir / f"{saved_idx:06d}.jpg"
                if not frame_path.exists():
                    cv2.imwrite(str(frame_path), frame)
                rows.append({"video_id": video_id, "t_sec": t_sec, "frame_path": str(frame_path)})
                saved_idx += 1
                pbar.update(1)
            frame_idx += 1
    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(index_path, index=False)
    return df, source_fps
