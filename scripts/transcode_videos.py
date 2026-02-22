"""Batch-transcode all dataset videos from FMP4 → H.264 for browser playback.

After transcoding, if frame predictions exist in the latest run output,
the state overlay (IDLE / TRANSIT / WORKING / DOWNTIME) is baked into
the video so the user sees it live in the Streamlit player.

Usage:
    python scripts/transcode_videos.py
    python scripts/transcode_videos.py --force   # re-transcode all
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
H264_DIR = ROOT / "outputs" / "cache" / "h264"
RUNS_DIR = ROOT / "outputs" / "runs" / "latest"

# ── State colors (BGR for OpenCV) ──
STATE_COLORS = {
    "IDLE":     (0x28, 0x27, 0xd6),   # red-ish  (#d62728)
    "TRANSIT":  (0xb4, 0x77, 0x1f),   # blue     (#1f77b4)
    "WORKING":  (0x2c, 0xa0, 0x2c),   # green    (#2ca02c)
    "DOWNTIME": (0x0e, 0x7f, 0xff),   # orange   (#ff7f0e)
}
STATES_ORDER = ["WORKING", "TRANSIT", "IDLE", "DOWNTIME"]


# ── Prediction loading ──

def _load_predictions(csv_path: Path) -> list[tuple[float, str]]:
    """Return sorted list of (t_sec, label_smoothed) from CSV."""
    rows: list[tuple[float, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["t_sec"])
            label = row.get("label_smoothed", row.get("label_raw", "IDLE"))
            rows.append((t, label))
    rows.sort(key=lambda r: r[0])
    return rows


def _get_label_at_time(predictions: list[tuple[float, str]], t: float) -> str:
    """Find the label for the nearest prediction time <= t."""
    best = predictions[0][1] if predictions else "IDLE"
    for pt, lbl in predictions:
        if pt > t + 0.05:
            break
        best = lbl
    return best


# ── Overlay drawing ──

def _draw_state_legend(frame: np.ndarray, active_label: str) -> np.ndarray:
    """Draw all 4 states in the top-right, highlighting the active one."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.2, w / 1200))
    thickness = max(1, int(font_scale * 2))
    thin = max(1, int(font_scale * 1.5))

    margin = int(w * 0.015)
    row_pad = int(6 * font_scale)
    dot_r = int(8 * font_scale)

    # Measure text sizes
    text_sizes = []
    for s in STATES_ORDER:
        (tw, th), bl = cv2.getTextSize(s, font, font_scale, thickness)
        text_sizes.append((tw, th, bl))
    max_tw = max(ts[0] for ts in text_sizes)
    max_th = max(ts[1] for ts in text_sizes)
    row_h = max_th + row_pad * 2

    # Panel dimensions
    panel_w = dot_r * 2 + 12 + max_tw + 2 * margin + 10
    panel_h = len(STATES_ORDER) * row_h + margin * 2

    # Position: top-right
    x1 = w - panel_w - margin
    y1 = margin

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x1 + panel_w, y1 + panel_h), (80, 80, 80), 1)

    for i, state in enumerate(STATES_ORDER):
        color = STATE_COLORS.get(state, (200, 200, 200))
        is_active = (state == active_label)

        row_y = y1 + margin + i * row_h
        tw, th, bl = text_sizes[i]

        # Dot
        cx = x1 + margin + dot_r
        cy = row_y + row_h // 2
        if is_active:
            cv2.circle(frame, (cx, cy), dot_r + 3, color, -1)
            cv2.circle(frame, (cx, cy), dot_r, (255, 255, 255), -1)
        else:
            cv2.circle(frame, (cx, cy), dot_r - 1, color, -1)

        # Text
        text_x = cx + dot_r + 8
        text_y = row_y + (row_h + th) // 2
        if is_active:
            cv2.putText(frame, state, (text_x, text_y), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, state, (text_x, text_y), font, font_scale * 0.9,
                        (140, 140, 140), thin, cv2.LINE_AA)

    return frame


# ── Pre-render overlay panels (done once per video, not per frame) ──

def _make_state_legend_panels(w: int, h: int) -> dict[str, np.ndarray]:
    """Pre-render a transparent legend panel for each state.
    Returns a dict of label → BGRA overlay (same size as frame)."""
    panels = {}
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    for state in STATES_ORDER:
        canvas = dummy.copy()
        _draw_state_legend(canvas, state)
        # Store as BGRA with diff on top of black — we'll blend per frame
        panels[state] = canvas
    return panels


# ── Transcode with overlay ──

def _try_nvenc() -> bool:
    """Check whether ffmpeg has h264_nvenc available."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True
    )
    return "h264_nvenc" in result.stdout


def _build_ffmpeg_cmd(w: int, h: int, fps: float, dst: Path, use_nvenc: bool) -> list[str]:
    encoder = ["h264_nvenc", "-preset", "p4"] if use_nvenc else ["libx264", "-preset", "fast", "-crf", "23"]
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", *encoder,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]


def _transcode_with_overlay(src: Path, dst: Path, predictions: list[tuple[float, str]]) -> bool:
    """Read src frame-by-frame, draw overlay, pipe raw frames to ffmpeg for H.264."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"FAILED — could not open {src}")
        return False

    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pre-render legend panels once
    panels = _make_state_legend_panels(w, h)

    use_nvenc = _try_nvenc()
    enc_label = "NVENC" if use_nvenc else "libx264"
    ffmpeg_cmd = _build_ffmpeg_cmd(w, h, fps, dst, use_nvenc)

    # stderr=DEVNULL avoids the pipe-buffer deadlock that caused it to hang
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    pbar = tqdm(total=total, desc=f"  {src.name} [{enc_label}]", unit="frame", leave=True)
    prev_label = None
    panel = None
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        label = _get_label_at_time(predictions, t)

        # Only re-blend when state changes (skip redundant work)
        if label != prev_label:
            panel = panels[label].copy()
            prev_label = label

        # Blend pre-rendered panel over frame where panel != black
        mask = (panel.sum(axis=2) > 0)
        frame[mask] = panel[mask]

        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            break
        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    proc.stdin.close()
    proc.wait()   # safe now — stderr is DEVNULL so no buffer deadlock

    if idx == 0 or proc.returncode != 0:
        print(f"FAILED (exit {proc.returncode}) — try running manually with ffmpeg")
        return False

    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"done ({idx} frames, {size_mb:.1f} MB, {enc_label}, overlay applied)")
    return True



def _transcode_plain(src: Path, dst: Path) -> bool:
    """Transcode via ffmpeg without overlay (fallback when no predictions)."""
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
        print(f"done ({size_mb:.1f} MB, no overlay)")
        return True
    else:
        print(f"FAILED (exit {result.returncode})")
        print(result.stderr.decode("utf-8", errors="replace")[-500:])
        return False


def _find_predictions_csv(video_stem: str) -> Path | None:
    """Look for a matching frame_predictions CSV in the latest run output."""
    if not RUNS_DIR.exists():
        return None
    csv_path = RUNS_DIR / f"{video_stem}_frame_predictions.csv"
    if csv_path.exists():
        return csv_path
    # Try glob in case naming differs slightly
    matches = list(RUNS_DIR.glob(f"*{video_stem}*frame_predictions.csv"))
    return matches[0] if matches else None


def transcode_all(force: bool = False) -> None:
    H264_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(DATASET_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {DATASET_DIR}")
        return

    print(f"Found {len(videos)} videos in {DATASET_DIR}")
    for i, src in enumerate(videos, 1):
        dst = H264_DIR / src.name
        if dst.exists() and not force:
            print(f"  [{i}/{len(videos)}] {src.name} — already transcoded, skipping")
            continue

        # Check for predictions
        pred_csv = _find_predictions_csv(src.stem)
        if pred_csv:
            predictions = _load_predictions(pred_csv)
            print(f"  [{i}/{len(videos)}] {src.name} — transcoding with overlay ({len(predictions)} predictions)...", end=" ", flush=True)
            _transcode_with_overlay(src, dst, predictions)
        else:
            print(f"  [{i}/{len(videos)}] {src.name} — transcoding (no predictions found)...", end=" ", flush=True)
            _transcode_plain(src, dst)

    print(f"\n✓ All H.264 videos are in {H264_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-transcode all videos even if they exist")
    args = parser.parse_args()
    transcode_all(force=args.force)
