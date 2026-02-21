from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from app.components import blocker_chart, canvas_timeline_html, metrics_table
from src.analytics import build_productivity_heatmap, generate_site_level_insights
from src.config import CONFIG
from src.pipeline import run_pipeline
from src.report import generate_daily_report, report_to_pdf

st.set_page_config(page_title="Construction Productivity Analyzer", layout="wide")


def load_outputs(run_dir: Path):
    required = [run_dir / "metrics.json", run_dir / "segments.json", run_dir / "rankings.json"]
    if not all(p.exists() for p in required):
        return None
    return {
        "metrics": json.loads((run_dir / "metrics.json").read_text()),
        "segments": json.loads((run_dir / "segments.json").read_text()),
        "rankings": json.loads((run_dir / "rankings.json").read_text()),
    }


def _find_frame_index(video_id: str) -> Path | None:
    base = CONFIG.cache_dir / "frames" / video_id
    if not base.exists():
        return None
    hits = sorted(base.glob("**/frame_index_*.csv"))
    return hits[-1] if hits else None


st.title("Egocentric Construction Productivity Analyzer")
controls = st.columns([1, 1, 1, 1])
if controls[0].button("Run / Refresh Pipeline"):
    with st.spinner("Running pipeline..."):
        run_pipeline(CONFIG.dataset_dir, CONFIG.run_dir, CONFIG.fps)
    st.rerun()

outputs = load_outputs(CONFIG.run_dir)
if outputs is None:
    st.info("No outputs found yet. Run the pipeline first.")
    st.stop()

metrics_df = pd.DataFrame(outputs["metrics"])
if metrics_df.empty:
    st.warning("No metrics available.")
    st.stop()

tasks = sorted(metrics_df["task"].unique().tolist())
selected_task = controls[1].selectbox("Task", tasks)
task_df = metrics_df[metrics_df["task"] == selected_task].copy().sort_values("video_id")
selected_video = controls[2].selectbox("Video", task_df["video_id"].tolist())
selected_row = task_df[task_df["video_id"] == selected_video].iloc[0].to_dict()
segments = outputs["segments"].get(selected_video, [])
max_t = int(max((s.get("end_sec", 0) for s in segments), default=1))
controls[3].selectbox("Run", [CONFIG.run_id], index=0)

# Pick up timeline clicks via query params
_qp = st.query_params
if "tl_click" in _qp:
    _clicked = int(_qp["tl_click"])
    st.session_state["slider_t"] = min(_clicked, max_t)
    st.query_params.clear()

overview_tab, rankings_tab, heatmap_tab, report_tab = st.tabs(["Overview", "Rankings", "Heatmap", "Daily Report"])

with overview_tab:
    left, right = st.columns([1.3, 1.0])
    with left:
        # Video player — load pre-transcoded H.264 from cache
        _h264_path = _project_root / "outputs" / "cache" / "h264" / f"{selected_video}.mp4"
        if _h264_path.exists():
            st.video(_h264_path.read_bytes())
        else:
            st.info("Video not transcoded yet. Run: `python scripts/transcode_videos.py`")

        _default_t = st.session_state.get("slider_t", 0)
        current_t = st.slider("Current time (sec)", 0, max_t, min(_default_t, max_t), key="slider_t")
        if segments:
            components.html(canvas_timeline_html(segments, current_t=current_t), height=90, scrolling=False)

        # ── Colored activity label ──
        active_label = ""
        for s in segments:
            if s["start_sec"] <= current_t < s["end_sec"]:
                active_label = s["label"]
                break
        _LABEL_STYLE = {
            "IDLE":    ("🔴", "#d62728", "#fff"),
            "TRANSIT": ("🔵", "#1f77b4", "#fff"),
            "WORKING": ("🟢", "#2ca02c", "#fff"),
        }
        if active_label and active_label in _LABEL_STYLE:
            icon, bg, fg = _LABEL_STYLE[active_label]
            st.markdown(
                f'<div style="background:{bg};color:{fg};padding:8px 16px;'
                f'border-radius:8px;font-size:18px;font-weight:700;'
                f'text-align:center;margin:4px 0 8px">'
                f'{icon} {active_label} @ {current_t}s</div>',
                unsafe_allow_html=True,
            )

        fi_path = _find_frame_index(selected_video)
        if fi_path and fi_path.exists():
            fi = pd.read_csv(fi_path)
            if not fi.empty:
                nearest = fi.iloc[(fi["t_sec"] - current_t).abs().argsort()[:1]]
                frame = Path(nearest.iloc[0]["frame_path"])
                if frame.exists():
                    st.image(str(frame), caption=f"Frame @ {nearest.iloc[0]['t_sec']:.1f}s")

    with right:
        st.subheader("Segments")
        sdf = pd.DataFrame(segments)
        if not sdf.empty:
            # Find active segment index
            active_idx = None
            for _i, _s in enumerate(segments):
                if _s["start_sec"] <= current_t < _s["end_sec"]:
                    active_idx = _i
                    break

            display_cols = ["label", "start_sec", "end_sec", "duration_sec", "confidence"]
            show_df = sdf[display_cols].copy()

            # Color-highlight the active row
            _row_colors = {"IDLE": "#d62728", "TRANSIT": "#1f77b4", "WORKING": "#2ca02c"}
            def _highlight_active(row):
                idx = row.name
                if idx == active_idx:
                    bg = _row_colors.get(segments[idx]["label"], "#333")
                    return [f"background-color: {bg}; color: #fff; font-weight: bold"] * len(row)
                return [""] * len(row)

            styled = show_df.style.apply(_highlight_active, axis=1)
            st.dataframe(styled, height=280, use_container_width=True)
        st.subheader("Summary")
        st.table(metrics_table(selected_row))

        st.subheader("Blocker insights")
        blockers = pd.DataFrame(selected_row.get("idle_blockers", []))
        if blockers.empty:
            st.info("No idle blockers detected (or VLM unavailable).")
        else:
            st.dataframe(
                blockers[["start_sec", "end_sec", "blocker_category", "confidence", "recommended_action"]],
                height=210,
                use_container_width=True,
            )
            fig = blocker_chart(selected_row.get("idle_blocker_summary", {}))
            if fig:
                st.plotly_chart(fig, use_container_width=True)

with rankings_tab:
    ranks = outputs["rankings"].get(selected_task, [])
    rdf = pd.DataFrame(ranks)
    if rdf.empty:
        st.info("No rankings.")
    else:
        st.dataframe(rdf[["video_id", "person_id", "score", "top_factors", "working_pct", "idle_pct"]])
    for insight in generate_site_level_insights(outputs["metrics"]):
        st.markdown(f"- {insight}")

with heatmap_tab:
    heat = build_productivity_heatmap(outputs["metrics"], outputs["segments"], CONFIG.day_start_time)
    if heat.empty:
        st.info("No heatmap data.")
    else:
        pivot = heat.pivot(index="worker", columns="hour", values="working_pct").fillna(0)
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlGn", labels={"color": "Working %"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(heat[["worker", "hour", "working_sec", "idle_sec", "transit_sec"]], height=200)

with report_tab:
    report_md = generate_daily_report(
        outputs["metrics"], outputs["rankings"], CONFIG.report_llm_provider, CONFIG.report_model
    )
    st.markdown(report_md)
    st.download_button(
        "Download PDF",
        data=report_to_pdf(report_md),
        file_name="daily_productivity_report.pdf",
        mime="application/pdf",
    )
