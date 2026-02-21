from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd


def _sum_dur(segments: list[dict], label: str) -> float:
    return sum(s["duration_sec"] for s in segments if s["label"] == label)


def video_metrics(video_id: str, person_id: str, task: str, segments: list[dict], idle_burst_thr: float) -> dict:
    total = max(sum(s["duration_sec"] for s in segments), 1e-8)
    working = _sum_dur(segments, "WORKING")
    idle = _sum_dur(segments, "IDLE")
    transit = _sum_dur(segments, "TRANSIT")

    transitions = max(len(segments) - 1, 0)
    transitions_per_min = transitions / (total / 60.0)

    idle_bursts = [s for s in segments if s["label"] == "IDLE" and s["duration_sec"] >= idle_burst_thr]
    idle_burst_count = len(idle_bursts)
    idle_burst_total = sum(s["duration_sec"] for s in idle_bursts)

    return {
        "video_id": video_id,
        "person_id": person_id,
        "task": task,
        "total_time_sec": total,
        "working_pct": working / total,
        "idle_pct": idle / total,
        "transit_pct": transit / total,
        "transitions_per_min": transitions_per_min,
        "idle_burst_count": idle_burst_count,
        "idle_burst_total_sec": idle_burst_total,
    }


def score_metric(m: dict, w: dict) -> tuple[float, dict]:
    breakdown = {
        "working": w["w_working"] * m["working_pct"],
        "idle_penalty": -w["w_idle"] * m["idle_pct"],
        "transit_penalty": -w["w_transit"] * m["transit_pct"],
        "transition_penalty": -w["w_transitions"] * m["transitions_per_min"],
        "idle_burst_penalty": -w["w_idle_bursts"] * (m["idle_burst_count"] + m["idle_burst_total_sec"] / 60.0),
    }
    return sum(breakdown.values()), breakdown


def build_rankings(metrics: list[dict], w: dict) -> dict:
    by_task = defaultdict(list)
    for m in metrics:
        enriched = dict(m)
        score, breakdown = score_metric(m, w)
        enriched["score"] = score
        enriched["score_breakdown"] = breakdown
        factors = sorted(breakdown.items(), key=lambda kv: abs(kv[1]), reverse=True)
        enriched["top_factors"] = [name for name, _ in factors[:3]]
        by_task[m["task"]].append(enriched)

    rankings = {}
    for task, rows in by_task.items():
        rankings[task] = sorted(rows, key=lambda x: x["score"], reverse=True)
    return rankings


def generate_insights(metric: dict) -> list[str]:
    insights = []
    if metric["transit_pct"] > 0.30:
        insights.append("High transit share suggests layout/tool staging inefficiency.")
    if metric["transitions_per_min"] > 6:
        insights.append("Frequent transitions indicate workflow fragmentation.")
    if metric["idle_burst_count"] > 0:
        insights.append("Long idle bursts may indicate blocking dependencies.")
    if metric["working_pct"] > 0.65:
        insights.append("Strong working share indicates sustained task engagement.")
    blocker_summary = metric.get("idle_blocker_summary", {})
    if blocker_summary.get("counts_by_category"):
        most_common = Counter(blocker_summary["counts_by_category"]).most_common(1)[0][0]
        insights.append(f"Most frequent blocker: {most_common}.")
    return insights


def generate_site_level_insights(metrics: list[dict]) -> list[str]:
    category_counter: Counter = Counter()
    for metric in metrics:
        for cat, count in metric.get("idle_blocker_summary", {}).get("counts_by_category", {}).items():
            category_counter[cat] += count
    if not category_counter:
        return []
    top_cat, top_count = category_counter.most_common(1)[0]
    return [f"Site-level pattern: '{top_cat}' appears {top_count} times across workers."]


def build_productivity_heatmap(metrics: list[dict], segments_by_video: dict, day_start_time: str = "07:00") -> pd.DataFrame:
    start_dt = datetime.strptime(day_start_time, "%H:%M")
    rows = []
    for metric in metrics:
        vid = metric["video_id"]
        segs = segments_by_video.get(vid, [])
        worker = metric.get("person_id", vid)
        hourly: dict[int, dict[str, float]] = defaultdict(lambda: {"WORKING": 0.0, "IDLE": 0.0, "TRANSIT": 0.0})
        for seg in segs:
            seg_start = float(seg["start_sec"])
            seg_end = float(seg["end_sec"])
            label = seg["label"]
            cursor = seg_start
            while cursor < seg_end:
                hour_idx = int(cursor // 3600)
                hour_end = min(seg_end, (hour_idx + 1) * 3600)
                hourly[hour_idx][label] += hour_end - cursor
                cursor = hour_end

        for hour_idx, vals in hourly.items():
            total = max(vals["WORKING"] + vals["IDLE"] + vals["TRANSIT"], 1e-8)
            bucket_time = (start_dt + timedelta(hours=hour_idx)).strftime("%H:00")
            rows.append(
                {
                    "worker": worker,
                    "video_id": vid,
                    "hour": bucket_time,
                    "working_pct": vals["WORKING"] / total,
                    "idle_pct": vals["IDLE"] / total,
                    "transit_pct": vals["TRANSIT"] / total,
                    "working_sec": vals["WORKING"],
                    "idle_sec": vals["IDLE"],
                    "transit_sec": vals["TRANSIT"],
                }
            )
    return pd.DataFrame(rows)
