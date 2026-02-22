from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import CONFIG, Config
from .labels import LABELS
from .vlm import vlm_infer


def _prompt_hash() -> str:
    prompt = _build_prompt()
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _cache_path(video_id: str) -> Path:
    path = CONFIG.cache_dir / "refined"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{video_id}.jsonl"


def _load_cache(video_id: str, model_name: str, prompt_hash: str) -> dict[float, dict[str, Any]]:
    cache: dict[float, dict[str, Any]] = {}
    cache_file = _cache_path(video_id)
    if not cache_file.exists():
        return cache
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("model_name") != model_name or item.get("prompt_hash") != prompt_hash:
            continue
        cache[float(item["t_sec"])] = item
    return cache


def _append_cache(video_id: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cache_file = _cache_path(video_id)
    with cache_file.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_prompt() -> str:
    labels_json = json.dumps(LABELS, indent=2)
    return (
        "You are a construction-activity visual labeling assistant.\n"
        "Analyze the frame and return STRICT JSON only.\n"
        "Choose exactly one refined_label from the allowed keys.\n"
        "Ground your answer in visible evidence only.\n"
        "If image is obstructed/ambiguous, return refined_label='UNCERTAIN'.\n\n"
        "Allowed labels and trigger schema:\n"
        f"{labels_json}\n\n"
        "Output schema:\n"
        "{\n"
        '  "refined_label": "<one key from allowed labels>",\n'
        '  "trigger_evidence": {\n'
        '    "hand_positions": ["..."],\n'
        '    "interaction_world": ["..."],\n'
        '    "movement": ["..."],\n'
        '    "camera_orientation": ["..."]\n'
        "  },\n"
        '  "confidence": 0.0,\n'
        '  "short_reason": "one sentence"\n'
        "}\n"
    )


def _parse_response(raw: str) -> dict[str, Any]:
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        payload = json.loads(raw[start : end + 1]) if start != -1 and end != -1 else json.loads(raw)
        refined = payload.get("refined_label", "UNCERTAIN")
        if refined not in LABELS:
            refined = "UNCERTAIN"
        return {
            "refined_label": refined,
            "trigger_evidence": payload.get("trigger_evidence", {}),
            "confidence": float(payload.get("confidence", 0.0)),
            "short_reason": str(payload.get("short_reason", ""))[:240],
            "raw_response": raw,
        }
    except Exception:
        return {
            "refined_label": "UNCERTAIN",
            "trigger_evidence": {},
            "confidence": 0.0,
            "short_reason": "Failed to parse VLM JSON.",
            "raw_response": raw,
        }


def _is_uncertain_probs(prob_row: np.ndarray) -> bool:
    arr = np.sort(np.asarray(prob_row, dtype=float))[::-1]
    max_prob = arr[0]
    second = arr[1] if len(arr) > 1 else 0.0
    return max_prob < 0.55 or (max_prob - second) < 0.15


def _select_frames(frame_preds: pd.DataFrame, call_policy: str) -> pd.Series:
    coarse = frame_preds["coarse_label_raw"].astype(str)
    uncertain = frame_preds.apply(
        lambda r: _is_uncertain_probs(np.array([r.get("p_idle", 0.0), r.get("p_working", r.get("p_work", 0.0)), r.get("p_transit", 0.0)])),
        axis=1,
    )
    if call_policy == "all":
        return pd.Series(True, index=frame_preds.index)
    if call_policy == "uncertain_only":
        return uncertain

    # idle_and_pause default
    motion_low = frame_preds.get("diff_energy", pd.Series(0.0, index=frame_preds.index)).fillna(0.0) < 0.02
    working = coarse == "WORKING"
    pause_candidate = working & motion_low
    return (coarse == "IDLE") | (working & uncertain) | pause_candidate


def _default_refined_for_coarse(coarse_label: str) -> str:
    return {
        "IDLE": "IDLE_WAITING",
        "WORKING": "WORKING",
        "TRANSIT": "TRANSIT",
        "DOWNTIME": "BREAK",
    }.get(coarse_label, "UNCERTAIN")


def infer_refined_labels(video_id: str, frame_preds: pd.DataFrame, config: Config = CONFIG) -> pd.DataFrame:
    model_name = config.refined_vlm_model
    prompt = _build_prompt()
    prompt_hash = _prompt_hash()
    call_policy = config.refined_call_policy
    max_frames = config.refined_max_frames_per_video

    out = frame_preds[["t_sec", "frame_path", "coarse_label_raw"]].copy()
    out["refined_label"] = out["coarse_label_raw"].map(_default_refined_for_coarse)
    out["refined_confidence"] = 0.0
    out["refined_reason"] = "Derived from coarse state."
    out["trigger_evidence"] = "{}"

    if config.refined_infer_mode == "all":
        selected_mask = _select_frames(frame_preds, "all")
    else:
        selected_mask = _select_frames(frame_preds, call_policy)

    selected_idx = frame_preds[selected_mask].index[:max_frames]
    if len(selected_idx) == 0:
        return out

    cache = _load_cache(video_id, model_name, prompt_hash)
    new_rows: list[dict[str, Any]] = []

    for i in selected_idx:
        t_sec = float(frame_preds.at[i, "t_sec"])
        cached = cache.get(t_sec)
        if cached is None:
            try:
                image = Image.open(frame_preds.at[i, "frame_path"]).convert("RGB")
                parsed = _parse_response(
                    vlm_infer(
                        [image],
                        prompt,
                        model_name=config.refined_vlm_model,
                        use_4bit=config.refined_vlm_4bit,
                        max_new_tokens=config.refined_max_new_tokens,
                        temperature=config.refined_temperature,
                    )
                )
            except Exception as exc:
                parsed = {
                    "refined_label": "UNCERTAIN",
                    "trigger_evidence": {},
                    "confidence": 0.0,
                    "short_reason": f"VLM unavailable: {exc}",
                    "raw_response": "",
                }
            cached = {
                "video_id": video_id,
                "t_sec": t_sec,
                "model_name": model_name,
                "prompt_hash": prompt_hash,
                "refined_label": parsed["refined_label"],
                "confidence": float(parsed["confidence"]),
                "short_reason": parsed["short_reason"],
                "trigger_evidence": parsed["trigger_evidence"],
                "raw_response": parsed.get("raw_response", ""),
            }
            new_rows.append(cached)

        out.at[i, "refined_label"] = cached.get("refined_label", "UNCERTAIN")
        out.at[i, "refined_confidence"] = float(cached.get("confidence", 0.0))
        out.at[i, "refined_reason"] = cached.get("short_reason", "")
        out.at[i, "trigger_evidence"] = json.dumps(cached.get("trigger_evidence", {}))

    _append_cache(video_id, new_rows)
    return out


def add_refined_segment_details(segments: list[dict], per_frame: pd.DataFrame) -> list[dict]:
    if not segments:
        return segments

    rows = []
    for seg in segments:
        mask = (per_frame["t_sec"] >= seg["start_sec"]) & (per_frame["t_sec"] < seg["end_sec"])
        slice_df = per_frame[mask]
        counter = Counter(slice_df["refined_label"].tolist())
        total = max(sum(counter.values()), 1)
        top2 = counter.most_common(2)
        seg["dominant_refined_label"] = top2[0][0] if top2 else "UNCERTAIN"
        seg["refined_top2"] = [[lbl, round(cnt / total, 4)] for lbl, cnt in top2]
        notes = [r for r in slice_df.get("refined_reason", pd.Series(dtype=str)).tolist() if r and r != "Derived from coarse state."]
        seg["refined_notes"] = notes[0] if notes else ""
        rows.append(seg)
    return rows


def refined_summary_from_frames(per_frame: pd.DataFrame) -> dict[str, Any]:
    counts = Counter(per_frame.get("refined_label", pd.Series(dtype=str)).tolist())
    times = {k: int(v) for k, v in counts.items()}
    blockers = {
        k: int(v)
        for k, v in counts.items()
        if k in {"IDLE_WAITING", "WORKING_PAUSE", "IDLE_DEVICE", "UNSAFE_BEHAVIOR", "UNCERTAIN"}
    }
    return {
        "counts_by_refined_label": dict(counts),
        "time_sec_by_refined_label": times,
        "top_blockers_inferred": sorted(blockers.items(), key=lambda kv: kv[1], reverse=True)[:3],
    }
