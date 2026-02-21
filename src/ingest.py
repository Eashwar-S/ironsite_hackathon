from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re

import cv2
import pandas as pd


VIDEO_EXTS = {".mp4", ".mov", ".mkv"}


@dataclass
class VideoRecord:
    video_id: str
    person_id: str
    task: str
    path: str
    duration_sec: float


def _duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames / fps) if fps > 0 else 0.0


def parse_video_name(path: Path, fallback_person_index: int) -> tuple[str, str, str]:
    video_id = path.stem
    m = re.match(r"^(\d+)_([A-Za-z0-9_\-]+)$", video_id)
    if m:
        numeric_prefix = m.group(1)
        task = m.group(2)
        person_id = f"Person {int(numeric_prefix):02d}"
    else:
        parts = video_id.split("_", 1)
        task = parts[1] if len(parts) > 1 else "unknown_task"
        person_id = f"Person {fallback_person_index:02d}"
    return video_id, person_id, task


def build_manifest(dataset_dir: Path, out_csv: Path) -> pd.DataFrame:
    files = sorted(
        [p for p in dataset_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    )

    records: list[VideoRecord] = []
    for idx, video_path in enumerate(files, start=1):
        video_id, person_id, task = parse_video_name(video_path, idx)
        records.append(
            VideoRecord(
                video_id=video_id,
                person_id=person_id,
                task=task,
                path=str(video_path),
                duration_sec=_duration(video_path),
            )
        )

    df = pd.DataFrame([asdict(r) for r in records])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
