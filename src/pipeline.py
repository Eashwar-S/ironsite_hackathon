from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .analytics import build_rankings, generate_insights, video_metrics
from .blockers import detect_idle_blockers
from .config import CONFIG
from .features import _device, compute_embeddings, compute_text_features
from .frames import extract_video_frames
from .ingest import build_manifest
from .model import DEFAULT_LABEL_PROMPTS, LABELS, zero_shot_classify
from .motion import compute_motion_features
from .report import generate_daily_report
from .smoothing import majority_smooth, to_segments

_AUTO_FPS_DIVISOR = 4  # sample at 1/4 of native fps when auto-detecting


def _detect_source_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return src


def _choose_fps(requested_fps: float, manifest: pd.DataFrame) -> float:
    """Return the effective fps to use.

    If the requested fps equals the config default (1.0) it is treated as
    'not explicitly set by the user', so we auto-detect from the first video.
    Otherwise the requested value is used as-is.
    """
    if requested_fps != 1.0:
        return requested_fps

    if manifest.empty:
        return requested_fps

    first_path = Path(manifest.iloc[0]["path"])
    if not first_path.exists():
        return requested_fps

    source_fps = _detect_source_fps(first_path)
    auto = max(1.0, round(source_fps / _AUTO_FPS_DIVISOR))
    print(
        f"[auto-fps] source video is {source_fps:.2f} fps → "
        f"sampling at {auto:.0f} fps (1/{_AUTO_FPS_DIVISOR} of native)"
    )
    return auto


_MAX_RECOMMENDED_FPS = 10.0


def run_pipeline(dataset: Path, out: Path, fps: float | None = None) -> None:
    fps = fps if fps is not None else CONFIG.fps

    device = _device()
    print(f"[device] {device}")

    cache_frames = CONFIG.cache_dir / "frames"
    cache_embeds = CONFIG.cache_dir / "embeds"
    cache_motion = CONFIG.cache_dir / "motion"
    for p in [cache_frames, cache_embeds, cache_motion, out]:
        p.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(dataset, out / "manifest.csv")

    # Auto-fps: only kicks in when fps was not explicitly overridden by user
    fps = _choose_fps(fps, manifest)
    fps_tag = f"fps{fps:.4f}"

    if fps > _MAX_RECOMMENDED_FPS:
        print(
            f"[warn] fps={fps:.0f} is high — expect large frame counts and slow embedding. "
            f"Recommended: ≤{_MAX_RECOMMENDED_FPS:.0f} fps."
        )

    # Compute text features once for all videos
    label_prompts = [DEFAULT_LABEL_PROMPTS[lbl] for lbl in LABELS]
    text_features = compute_text_features(
        label_prompts,
        model_name=CONFIG.model_name,
        fallback_model_name=CONFIG.fallback_model_name,
        device=device,
    )

    all_segments: dict[str, list[dict]] = {}
    all_metrics: list[dict] = []

    video_iter = tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Videos",
        unit="vid",
    )
    for _, row in video_iter:
        video_id = row["video_id"]
        task = row["task"]
        person_id = row["person_id"]
        video_path = Path(row["path"])

        video_iter.set_postfix({"video": video_id, "stage": "frames"})

        # fps-keyed subdirectory so different sampling rates don't share extracted frames
        frame_dir = cache_frames / video_id / fps_tag
        frame_index, _ = extract_video_frames(video_id, video_path, frame_dir, fps=fps)
        if frame_index.empty:
            all_segments[video_id] = []
            continue

        # fps-keyed cache filenames for embeddings and motion
        embed_file = cache_embeds / f"{video_id}_{fps_tag}.npz"
        video_iter.set_postfix({"video": video_id, "stage": "embedding", "frames": len(frame_index)})
        embeddings = compute_embeddings(
            frame_paths=frame_index["frame_path"].tolist(),
            cache_file=embed_file,
            model_name=CONFIG.model_name,
            fallback_model_name=CONFIG.fallback_model_name,
            batch_size=CONFIG.batch_size,
            device=device,
        )

        motion_file = cache_motion / f"{video_id}_{fps_tag}.csv"
        video_iter.set_postfix({"video": video_id, "stage": "motion"})
        motion_df = compute_motion_features(frame_index, motion_file)
        video_iter.set_postfix({"video": video_id, "stage": "classifying"})

        merged = frame_index.merge(motion_df[["t_sec", "diff_energy"]], on="t_sec", how="left")
        merged["diff_energy"] = merged["diff_energy"].fillna(0.0)

        # Guard: embeddings and frames must agree in length (they always should
        # now that caches are fps-keyed, but be defensive).
        n_frames = len(merged)
        n_embeds = len(embeddings)
        if n_embeds != n_frames:
            tqdm.write(
                f"[warn] {video_id}: embeddings ({n_embeds}) vs frames ({n_frames}) "
                "mismatch — truncating to shorter. Delete outputs/cache to regenerate."
            )
            n = min(n_embeds, n_frames)
            embeddings = embeddings[:n]
            merged = merged.iloc[:n].reset_index(drop=True)

        pred, probs = zero_shot_classify(
            embeddings, text_features, logit_scale=CONFIG.clip_logit_scale
        )

        smooth = majority_smooth(pred, CONFIG.smoothing_window)
        segments = to_segments(merged["t_sec"].to_numpy(), smooth, probs)

        all_segments[video_id] = segments

        metric = video_metrics(video_id, person_id, task, segments, CONFIG.idle_burst_seconds)
        try:
            blocker_data = detect_idle_blockers(video_id, segments, frame_index, CONFIG)
            metric["idle_blockers"] = blocker_data["idle_blockers"]
            metric["idle_blocker_summary"] = blocker_data["idle_blocker_summary"]
        except Exception as exc:
            tqdm.write(f"[warn] blocker detection skipped for {video_id}: {exc}")
            metric["idle_blockers"] = []
            metric["idle_blocker_summary"] = {
                "counts_by_category": {},
                "total_idle_seconds_explained": 0,
                "top_recommendation": "",
            }
        metric["insights"] = generate_insights(metric)
        all_metrics.append(metric)

        prob_df = pd.DataFrame(probs, columns=[f"p_{l.lower()}" for l in LABELS])
        per_frame = pd.concat(
            [merged[["video_id", "t_sec", "frame_path", "diff_energy"]], prob_df], axis=1
        )
        per_frame["label_raw"] = [LABELS[i] for i in pred]
        per_frame["label_smoothed"] = [LABELS[i] for i in smooth]
        per_frame.to_csv(out / f"{video_id}_frame_predictions.csv", index=False)

    weights = {
        "w_working": CONFIG.w_working,
        "w_idle": CONFIG.w_idle,
        "w_transit": CONFIG.w_transit,
        "w_transitions": CONFIG.w_transitions,
        "w_idle_bursts": CONFIG.w_idle_bursts,
    }
    rankings = build_rankings(all_metrics, weights)

    with open(out / "segments.json", "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=2)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    with open(out / "rankings.json", "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)

    report_md = generate_daily_report(
        all_metrics,
        rankings,
        provider=CONFIG.report_llm_provider,
        model=CONFIG.report_model,
    )
    (out / "daily_report.md").write_text(report_md, encoding="utf-8")

    tqdm.write(f"\n✓ Pipeline complete → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construction productivity pipeline")
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    parser.add_argument("--out", type=Path, default=CONFIG.run_dir)
    parser.add_argument(
        "--fps",
        type=float,
        default=CONFIG.fps,
        help="Extraction fps. Omit (or set to 1.0 default) to auto-detect from video.",
    )
    args = parser.parse_args()

    run_pipeline(args.dataset, args.out, args.fps)


if __name__ == "__main__":
    main()
