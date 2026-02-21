from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd
from PIL import Image

from .config import Config
from .vlm import vlm_infer

_BLOCKER_PROMPT = """You are an operations analyst for construction productivity. Answer ONLY in JSON.
These are frames from an egocentric camera during an IDLE burst.
Question: What is preventing this worker from being productive?
Return JSON:
{
  \"blocker_category\": one of [
    \"Waiting for materials\",
    \"Looking for tools\",
    \"Coordination bottleneck (watching another trade)\",
    \"Environment/weather stoppage\",
    \"Safety/inspection hold\",
    \"Unclear/Other\"
  ],
  \"evidence\": short string describing what you see,
  \"confidence\": float 0..1,
  \"recommended_action\": short suggestion
}
"""

_ALLOWED = {
    "Waiting for materials",
    "Looking for tools",
    "Coordination bottleneck (watching another trade)",
    "Environment/weather stoppage",
    "Safety/inspection hold",
    "Unclear/Other",
}


def _sample_idle_frames(segment: dict, frame_index: pd.DataFrame, max_frames: int) -> pd.DataFrame:
    start = float(segment["start_sec"])
    end = float(segment["end_sec"])
    in_seg = frame_index[(frame_index["t_sec"] >= start) & (frame_index["t_sec"] <= end)].copy()
    if in_seg.empty:
        return in_seg
    if len(in_seg) <= max_frames:
        return in_seg
    picks = [round(i * (len(in_seg) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return in_seg.iloc[picks]


def _parse_blocker_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "blocker_category": "Unclear/Other",
            "evidence": raw[:300],
            "confidence": 0.1,
            "recommended_action": "Review sampled frames manually.",
            "raw_response": raw,
        }

    category = parsed.get("blocker_category", "Unclear/Other")
    if category not in _ALLOWED:
        category = "Unclear/Other"

    confidence = parsed.get("confidence", 0.2)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence = 0.2

    return {
        "blocker_category": category,
        "evidence": str(parsed.get("evidence", ""))[:400],
        "confidence": confidence,
        "recommended_action": str(parsed.get("recommended_action", "Review manually."))[:240],
    }


def detect_idle_blockers(video_id: str, segments: list[dict], frame_index: pd.DataFrame, config: Config) -> dict:
    idle_segments = [
        s for s in segments if s.get("label") == "IDLE" and float(s.get("duration_sec", 0.0)) >= config.idle_burst_sec
    ]
    blockers: list[dict[str, Any]] = []

    for seg in idle_segments:
        sampled = _sample_idle_frames(seg, frame_index, config.blocker_max_frames)
        if sampled.empty:
            continue
        images = [Image.open(path).convert("RGB") for path in sampled["frame_path"].tolist()]
        try:
            raw = vlm_infer(images, _BLOCKER_PROMPT)
            parsed = _parse_blocker_json(raw)
        finally:
            for img in images:
                img.close()

        blockers.append(
            {
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "duration_sec": seg["duration_sec"],
                **parsed,
                "frames_used": sampled["t_sec"].round(2).tolist(),
            }
        )

    counts = Counter(b["blocker_category"] for b in blockers)
    total_explained = sum(float(b["duration_sec"]) for b in blockers)
    top_reco = ""
    if blockers:
        top_reco = Counter(b["recommended_action"] for b in blockers).most_common(1)[0][0]

    summary = {
        "counts_by_category": dict(counts),
        "total_idle_seconds_explained": total_explained,
        "top_recommendation": top_reco,
    }
    return {"idle_blockers": blockers, "idle_blocker_summary": summary, "video_id": video_id}
