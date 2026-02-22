from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from io import BytesIO
from typing import Any

import pandas as pd
from PIL import Image

from .config import Config
from .vlm import vlm_infer

# Design note for maintainers:
# - Frame sampling now combines deterministic anchors (start/middle/end), optional motion peaks
#   from `diff_energy`, and light uniform fill to maximize useful evidence per VLM call.
# - We reduce token and image cost by default (fewer frames + resize before inference), and we
#   opportunistically reuse a recent segment result when near-duplicate idle segments appear.
# - Parsing is tolerant to malformed wrappers around JSON and is backward-compatible with older
#   outputs by filling defaults.
# - A sparse second-stage analysis only runs for low-confidence first-stage outputs so cost stays
#   bounded while still improving ambiguous cases.

_BLOCKER_PROMPT = """You are an operations analyst for construction productivity.
You will receive a few frames from an egocentric camera during one IDLE segment and segment metadata.
Output JSON ONLY. No markdown. No extra keys.
Do not invent facts not visible from images or metadata.
If uncertain, choose \"Unclear/Other\" and use `what_to_check_next` to request evidence needed.

Required JSON schema:
{
  "blocker_category": one of [
    "Waiting for materials",
    "Looking for tools",
    "Coordination bottleneck (watching another trade)",
    "Environment/weather stoppage",
    "Safety/inspection hold",
    "Unclear/Other"
  ],
  "blocker_subtype": "short subtype, optional but preferred, <=40 chars",
  "confidence": float 0..1,
  "evidence": ["up to 3 visual cues, <=12 words each"],
  "recommended_action": "one sentence <=22 words",
  "time_to_fix_estimate_min": one of [1, 5, 10, 20, 30, 60, "unknown"],
  "who_can_fix": one of ["Worker", "Foreman", "Materials/Logistics", "Safety Officer", "Other", "Unknown"],
  "operational_impact": "one sentence <=18 words",
  "what_to_check_next": ["up to 3 short checks"],
  "risk_flag": one of ["none", "safety", "schedule", "quality", "unknown"]
}
"""

_DEEP_ANALYSIS_PROMPT = """You are reviewing an ambiguous idle-segment blocker classification.
Based on the same frames and metadata, return JSON ONLY:
{
  "hypothesis_1": "short likely blocker explanation",
  "hypothesis_2": "second plausible explanation",
  "disambiguation_request": "what additional evidence/frame would disambiguate"
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

_ALLOWED_FIX = {"Worker", "Foreman", "Materials/Logistics", "Safety Officer", "Other", "Unknown"}
_ALLOWED_RISK = {"none", "safety", "schedule", "quality", "unknown"}
_ALLOWED_TTF = {1, 5, 10, 20, 30, 60, "unknown"}


def _clip_text(value: Any, max_len: int) -> str:
    return str(value or "").strip()[:max_len]


def _motion_band(in_seg: pd.DataFrame) -> str:
    if "diff_energy" not in in_seg.columns or in_seg["diff_energy"].dropna().empty:
        return "unknown"
    val = float(in_seg["diff_energy"].fillna(0).mean())
    if val < 0.06:
        return "low"
    if val < 0.2:
        return "medium"
    return "high"


def _build_segment_context(segment: dict, prev_label: str | None, next_label: str | None, motion_summary: str) -> str:
    return (
        "Segment metadata:\n"
        f"- start_sec: {float(segment.get('start_sec', 0.0)):.2f}\n"
        f"- end_sec: {float(segment.get('end_sec', 0.0)):.2f}\n"
        f"- duration_sec: {float(segment.get('duration_sec', 0.0)):.2f}\n"
        f"- prev_label: {prev_label or 'unknown'}\n"
        f"- next_label: {next_label or 'unknown'}\n"
        f"- motion_summary: {motion_summary}\n"
    )


def _sample_idle_frames(segment: dict, frame_index: pd.DataFrame, max_frames: int) -> pd.DataFrame:
    start = float(segment["start_sec"])
    end = float(segment["end_sec"])
    in_seg = frame_index[(frame_index["t_sec"] >= start) & (frame_index["t_sec"] <= end)].copy()
    if in_seg.empty:
        return in_seg
    if len(in_seg) <= max_frames:
        return in_seg

    picked_idx: list[int] = []

    # deterministic anchors: start/mid/end
    anchors = [0, len(in_seg) // 2, len(in_seg) - 1]
    picked_idx.extend(anchors)

    # motion peaks if available (likely informative moments)
    if "diff_energy" in in_seg.columns and in_seg["diff_energy"].notna().any():
        peaks = in_seg["diff_energy"].fillna(0).nlargest(min(3, len(in_seg))).index.tolist()
        picked_idx.extend(in_seg.index.get_indexer(peaks).tolist())

    # uniform fill for coverage
    if max_frames > 1:
        uniform = [round(i * (len(in_seg) - 1) / (max_frames - 1)) for i in range(max_frames)]
        picked_idx.extend(uniform)

    dedup = sorted({i for i in picked_idx if 0 <= i < len(in_seg)})
    if len(dedup) > max_frames:
        dedup = dedup[:max_frames]
    return in_seg.iloc[dedup]


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    snippet = match.group(0)
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _parse_blocker_json(raw: str) -> dict[str, Any]:
    parsed = _extract_json_object(raw)
    if parsed is None:
        return {
            "blocker_category": "Unclear/Other",
            "blocker_subtype": "",
            "evidence": [_clip_text(raw, 120)],
            "confidence": 0.1,
            "recommended_action": "Review sampled frames manually.",
            "time_to_fix_estimate_min": "unknown",
            "who_can_fix": "Unknown",
            "operational_impact": "Potential unresolved idle time.",
            "what_to_check_next": ["Review more nearby frames"],
            "risk_flag": "unknown",
            "raw_response": _clip_text(raw, 600),
        }

    category = parsed.get("blocker_category", "Unclear/Other")
    if category not in _ALLOWED:
        category = "Unclear/Other"

    confidence = parsed.get("confidence", 0.2)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence = 0.2

    evidence = parsed.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    evidence = [_clip_text(e, 120) for e in evidence if str(e).strip()][:3] or ["No clear visual cue captured"]

    checks = parsed.get("what_to_check_next", [])
    if isinstance(checks, str):
        checks = [checks]
    if not isinstance(checks, list):
        checks = []
    checks = [_clip_text(c, 90) for c in checks if str(c).strip()][:3]

    who_can_fix = parsed.get("who_can_fix", "Unknown")
    if who_can_fix not in _ALLOWED_FIX:
        who_can_fix = "Unknown"

    ttf = parsed.get("time_to_fix_estimate_min", "unknown")
    if isinstance(ttf, str) and ttf.isdigit():
        ttf = int(ttf)
    if ttf not in _ALLOWED_TTF:
        ttf = "unknown"

    risk_flag = str(parsed.get("risk_flag", "unknown")).strip().lower()
    if risk_flag not in _ALLOWED_RISK:
        risk_flag = "unknown"

    return {
        "blocker_category": category,
        "blocker_subtype": _clip_text(parsed.get("blocker_subtype", ""), 40),
        "evidence": evidence,
        "confidence": confidence,
        "recommended_action": _clip_text(parsed.get("recommended_action", "Review manually."), 220),
        "time_to_fix_estimate_min": ttf,
        "who_can_fix": who_can_fix,
        "operational_impact": _clip_text(parsed.get("operational_impact", "Operational impact uncertain."), 180),
        "what_to_check_next": checks,
        "risk_flag": risk_flag,
    }


def _parse_deep_analysis_json(raw: str) -> dict[str, str]:
    parsed = _extract_json_object(raw) or {}
    return {
        "hypothesis_1": _clip_text(parsed.get("hypothesis_1", ""), 140),
        "hypothesis_2": _clip_text(parsed.get("hypothesis_2", ""), 140),
        "disambiguation_request": _clip_text(parsed.get("disambiguation_request", ""), 180),
    }


def _frame_signature(sampled: pd.DataFrame) -> str:
    key = "|".join(f"{float(v):.2f}" for v in sampled["t_sec"].tolist())
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def _resize_image_for_vlm(img: Image.Image, long_side: int = 512) -> Image.Image:
    clone = img.copy()
    clone.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    if clone.mode != "RGB":
        clone = clone.convert("RGB")
    return clone


def _hash_images(images: list[Image.Image]) -> str:
    dig = hashlib.md5()
    for img in images:
        with BytesIO() as buff:
            img.save(buff, format="JPEG", quality=85)
            dig.update(buff.getvalue())
    return dig.hexdigest()


def _deterministic_video_summary(blockers: list[dict[str, Any]], total_idle_seconds: float, counts: Counter) -> str:
    if not blockers:
        return "No blocker events were analyzed for this video."
    top_category = counts.most_common(1)[0][0] if counts else "Unclear/Other"
    top_action = Counter(b.get("recommended_action", "") for b in blockers if b.get("recommended_action")).most_common(1)
    action = top_action[0][0] if top_action else "Review idle events manually"
    return (
        f"Analyzed {len(blockers)} idle bursts covering {total_idle_seconds:.1f}s. "
        f"Most frequent blocker was **{top_category}**. "
        f"Most common recommended action: **{action}**."
    )


def detect_idle_blockers(video_id: str, segments: list[dict], frame_index: pd.DataFrame, config: Config) -> dict:
    idle_segments = [
        s for s in segments if s.get("label") == "IDLE" and float(s.get("duration_sec", 0.0)) >= config.idle_burst_sec
    ]
    blockers: list[dict[str, Any]] = []
    max_frames = max(1, min(int(getattr(config, "blocker_max_frames", 8)), 8))
    last_sig: str | None = None
    last_end: float | None = None
    last_result: dict[str, Any] | None = None
    skip_window_sec = float(os.getenv("BLOCKER_REUSE_WINDOW_SEC", "12"))

    for idx, seg in enumerate(idle_segments):
        sampled = _sample_idle_frames(seg, frame_index, max_frames)
        if sampled.empty:
            continue

        prev_label = idle_segments[idx - 1].get("label") if idx > 0 else None
        next_label = idle_segments[idx + 1].get("label") if idx + 1 < len(idle_segments) else None
        motion_summary = _motion_band(sampled)
        prompt = f"{_BLOCKER_PROMPT}\n\n{_build_segment_context(seg, prev_label, next_label, motion_summary)}"

        images = [_resize_image_for_vlm(Image.open(path).convert("RGB"), long_side=512) for path in sampled["frame_path"].tolist()]
        sig = _frame_signature(sampled) + _hash_images(images)[:12]
        should_reuse = (
            last_result is not None
            and last_sig == sig
            and last_end is not None
            and (float(seg.get("start_sec", 0.0)) - last_end) <= skip_window_sec
        )

        if should_reuse:
            parsed = dict(last_result)
            parsed["inference_reused"] = True
        else:
            raw = vlm_infer(images, prompt)
            parsed = _parse_blocker_json(raw)
            parsed["inference_reused"] = False

            if parsed.get("confidence", 0.0) < 0.55:
                deep_raw = vlm_infer(images, f"{_DEEP_ANALYSIS_PROMPT}\n\n{_build_segment_context(seg, prev_label, next_label, motion_summary)}")
                parsed["low_confidence_notes"] = _parse_deep_analysis_json(deep_raw)

            last_sig = sig
            last_end = float(seg.get("end_sec", 0.0))
            last_result = dict(parsed)

        for img in images:
            img.close()

        blockers.append(
            {
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "duration_sec": seg["duration_sec"],
                **parsed,
                "frames_used": sampled["t_sec"].round(2).tolist(),
                "representative_frame_paths": sampled["frame_path"].head(3).tolist(),
            }
        )

    counts = Counter(b["blocker_category"] for b in blockers)
    total_explained = sum(float(b["duration_sec"]) for b in blockers)
    top_reco = ""
    if blockers:
        top_reco = Counter(b["recommended_action"] for b in blockers).most_common(1)[0][0]

    by_category: dict[str, dict[str, Any]] = {}
    for category in _ALLOWED:
        cat_events = [b for b in blockers if b.get("blocker_category") == category]
        if not cat_events:
            continue
        cat_duration = sum(float(e.get("duration_sec", 0.0)) for e in cat_events)
        cat_action = Counter(e.get("recommended_action", "") for e in cat_events if e.get("recommended_action")).most_common(1)
        cat_action_txt = cat_action[0][0] if cat_action else "Review events in this category"
        by_category[category] = {
            "count": len(cat_events),
            "total_duration_sec": round(cat_duration, 2),
            "insight": f"{len(cat_events)} events over {cat_duration:.1f}s indicate repeated {category.lower()} blockers.",
            "systemic_fix": cat_action_txt,
        }

    summary = {
        "counts_by_category": dict(counts),
        "total_idle_seconds_explained": total_explained,
        "idle_bursts_analyzed": len(blockers),
        "top_category": counts.most_common(1)[0][0] if counts else "",
        "top_recommendation": top_reco,
        "category_duration_sec": {k: round(sum(float(b.get("duration_sec", 0.0)) for b in blockers if b.get("blocker_category") == k), 2) for k in counts},
    }

    video_summary = _deterministic_video_summary(blockers, total_explained, counts)
    if os.getenv("BLOCKER_SUMMARY_LLM", "false").lower() in {"1", "true", "yes"} and blockers:
        summary_prompt = (
            "Create a short markdown narrative (3 short bullets max) summarizing blocker patterns using ONLY this structured JSON. "
            "No invented facts. Return JSON {\"video_summary_md\": str, \"category_insights\": object}.\n"
            + json.dumps({"summary": summary, "events": blockers[:40]})
        )
        blank = Image.new("RGB", (8, 8), color=(0, 0, 0))
        raw_summary = vlm_infer([blank], summary_prompt)
        blank.close()
        parsed_summary = _extract_json_object(raw_summary) or {}
        summary["video_summary_md"] = _clip_text(parsed_summary.get("video_summary_md", video_summary), 1200)
        summary["category_insights"] = parsed_summary.get("category_insights", by_category)
    else:
        summary["video_summary_md"] = video_summary
        summary["category_insights"] = by_category

    return {"idle_blockers": blockers, "idle_blocker_summary": summary, "video_id": video_id}
