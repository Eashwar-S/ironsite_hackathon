"""Batch-transcode all dataset videos from FMP4 → H.264 for browser playback.

Usage:
    python scripts/transcode_videos.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
H264_DIR = ROOT / "outputs" / "cache" / "h264"


def transcode_all() -> None:
    H264_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(DATASET_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {DATASET_DIR}")
        return

    print(f"Found {len(videos)} videos in {DATASET_DIR}")
    for i, src in enumerate(videos, 1):
        dst = H264_DIR / src.name
        if dst.exists():
            print(f"  [{i}/{len(videos)}] {src.name} — already transcoded, skipping")
            continue
        print(f"  [{i}/{len(videos)}] {src.name} — transcoding...", end=" ", flush=True)
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-c:a", "aac",
                "-movflags", "+faststart",
                str(dst),
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(f"done ({size_mb:.1f} MB)")
        else:
            print(f"FAILED (exit {result.returncode})")
            print(result.stderr.decode("utf-8", errors="replace")[-500:])

    print(f"\n✓ All H.264 videos are in {H264_DIR}")


if __name__ == "__main__":
    transcode_all()
