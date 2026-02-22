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

st.set_page_config(page_title="SiteSense AI | Construction Productivity", layout="wide")

# ── Global Styles ──
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] > div {
        font-size: 1.2rem !important;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    .main {
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)


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

        # Use session state to track time (updated by timeline clicks)
        current_t = st.session_state.get("slider_t", 0)
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
            "DOWNTIME": ("⚠️", "#ff7f0e", "#fff"),
        }
        # if active_label and active_label in _LABEL_STYLE:
        #     icon, bg, fg = _LABEL_STYLE[active_label]
        #     st.markdown(
        #         f'<div style="background:{bg};color:{fg};padding:8px 16px;'
        #         f'border-radius:8px;font-size:18px;font-weight:700;'
        #         f'text-align:center;margin:4px 0 8px">'
        #         f'{icon} {active_label} @ {current_t}s</div>',
        #         unsafe_allow_html=True,
        #     )

        # fi_path = _find_frame_index(selected_video)
        # if fi_path and fi_path.exists():
        #     fi = pd.read_csv(fi_path)
        #     if not fi.empty:
        #         nearest = fi.iloc[(fi["t_sec"] - current_t).abs().argsort()[:1]]
        #         frame = Path(nearest.iloc[0]["frame_path"])
        #         if frame.exists():
        #             st.image(str(frame), caption=f"Frame @ {nearest.iloc[0]['t_sec']:.1f}s")

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
            if "dominant_refined_label" in sdf.columns:
                show_df["label"] = show_df["label"] + " (Refined: " + sdf["dominant_refined_label"].fillna("UNCERTAIN") + ")"

            # Color-highlight the active row
            _row_colors = {"IDLE": "#d62728", "TRANSIT": "#1f77b4", "WORKING": "#2ca02c", "DOWNTIME": "#ff7f0e"}
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

        refined_summary = selected_row.get("refined_summary", {})
        refined_times = refined_summary.get("time_sec_by_refined_label", {}) if refined_summary else {}
        if refined_times:
            st.caption("Top refined labels")
            _rdf = pd.DataFrame(
                [{"refined_label": k, "duration_sec": v} for k, v in refined_times.items()]
            ).sort_values("duration_sec", ascending=False).head(5)
            st.table(_rdf)

        st.markdown("---")
        st.markdown("### 🔍 Blocker Insights")
        blocker_events = selected_row.get("idle_blockers", []) or []
        blockers = pd.DataFrame(blocker_events)
        summary = selected_row.get("idle_blocker_summary", {}) or {}

        # Shared Plotly dark template for consistency
        _PLOTLY_TEMPLATE = "plotly_dark"
        _CHART_MARGIN = dict(l=16, r=16, t=48, b=16)
        _CHART_FONT = dict(family="Inter, sans-serif", size=13)

        if blockers.empty:
            st.info("No idle blockers detected (or VLM unavailable).")
        else:
            fallback_counts = blockers["blocker_category"].value_counts().to_dict() if "blocker_category" in blockers.columns else {}
            counts = summary.get("counts_by_category", fallback_counts)
            seconds = summary.get("category_duration_sec", {})
            total_seconds = float(summary.get("total_idle_seconds_explained", blockers.get("duration_sec", pd.Series(dtype=float)).fillna(0).sum()))
            top_category = summary.get("top_category") or (max(counts, key=counts.get) if counts else "N/A")
            top_action = summary.get("top_recommendation") or (
                blockers["recommended_action"].mode().iloc[0] if "recommended_action" in blockers.columns and not blockers["recommended_action"].dropna().empty else "N/A"
            )
            summary_text = summary.get("video_summary_md") or (
                f"Analyzed **{len(blockers)}** idle burst(s) covering **{total_seconds:.1f}s**. "
                f"Most frequent blocker: **{top_category}**. "
                f"Most common recommended action: **{top_action}**."
            )

            st.markdown(summary_text)
            st.markdown("")

            # ── KPI row ──
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Idle Bursts", int(summary.get("idle_bursts_analyzed", len(blockers))))
            kpi2.metric("Time Explained", f"{total_seconds:.1f}s")
            kpi3.metric("Top Category", top_category)
            kpi4.metric("Top Action", top_action)

            st.markdown("")

            # ── Charts row: timeline + category breakdown side-by-side ──
            chart_left, chart_right = st.columns([1.4, 1.0])

            with chart_left:
                if "start_sec" in blockers.columns and "blocker_category" in blockers.columns:
                    tl_df = blockers.copy()
                    tl_df["duration_sec"] = tl_df.get("duration_sec", 1).fillna(1)
                    tl_fig = px.scatter(
                        tl_df,
                        x="start_sec",
                        y="blocker_category",
                        size="duration_sec",
                        color="blocker_category",
                        hover_data=[c for c in ["end_sec", "duration_sec", "confidence", "recommended_action"] if c in tl_df.columns],
                        title="⏱ Blocker Event Timeline",
                        template=_PLOTLY_TEMPLATE,
                    )
                    tl_fig.update_layout(
                        height=300,
                        margin=_CHART_MARGIN,
                        font=_CHART_FONT,
                        title_font_size=16,
                        legend_title_text="Category",
                        xaxis_title="Time (seconds)",
                        yaxis_title="",
                        showlegend=False,
                    )
                    tl_fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="#fff")))
                    st.plotly_chart(tl_fig, use_container_width=True)

            with chart_right:
                fig = blocker_chart(summary)
                if fig:
                    fig.update_layout(
                        template=_PLOTLY_TEMPLATE,
                        font=_CHART_FONT,
                        title_font_size=16,
                        title_text="📊 Blockers by Category",
                        height=300,
                        margin=_CHART_MARGIN,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # ── Per-category tabs ──
            st.markdown("#### Category Breakdown")
            category_tabs = st.tabs([
                "📦 Materials",
                "🔧 Tools",
                "👷 Coordination",
                "🛡️ Safety/Inspection",
                "❓ Other",
            ])
            category_map = [
                "Waiting for materials",
                "Looking for tools",
                "Coordination bottleneck (watching another trade)",
                "Safety/inspection hold",
                "Other",
            ]

            for tab, category in zip(category_tabs, category_map):
                with tab:
                    if "blocker_category" not in blockers.columns:
                        st.info("No category data available.")
                        continue
                    cat_df = blockers[blockers["blocker_category"] == category].copy()
                    if category == "Other" and cat_df.empty:
                        other_cats = {
                            "Environment/weather stoppage",
                            "Other",
                        }
                        cat_df = blockers[blockers["blocker_category"].isin(other_cats)].copy()

                    if cat_df.empty:
                        st.info("No events in this category.")
                        continue

                    cat_secs = float(cat_df.get("duration_sec", pd.Series(dtype=float)).fillna(0).sum())
                    cat_count = len(cat_df)
                    top_cat_actions = (
                        cat_df["recommended_action"].fillna("").value_counts().head(3).index.tolist()
                        if "recommended_action" in cat_df.columns
                        else []
                    )

                    h1, h2 = st.columns(2)
                    h1.metric("Events", cat_count)
                    h2.metric("Total Duration", f"{seconds.get(category, cat_secs):.1f}s")
                    if top_cat_actions:
                        actions_str = " **·** ".join(a for a in top_cat_actions if a)
                        st.markdown(f"**Recommended actions:** {actions_str}")

                    table_cols = [
                        "start_sec",
                        "end_sec",
                        "duration_sec",
                        "confidence",
                        "blocker_subtype",
                        "who_can_fix",
                        "time_to_fix_estimate_min",
                        "risk_flag",
                        "refined_context",
                    ]
                    avail_cols = [c for c in table_cols if c in cat_df.columns]
                    show_df = cat_df[avail_cols].sort_values("start_sec", ascending=True)
                    # Cast mixed-type columns to str to avoid Arrow serialization errors
                    for col in ["time_to_fix_estimate_min", "risk_flag", "who_can_fix", "blocker_subtype", "refined_context"]:
                        if col in show_df.columns:
                            show_df[col] = show_df[col].astype(str)
                    st.dataframe(show_df, use_container_width=True, height=180)

                    event_opts = [f"{i}: {row.start_sec:.1f}-{row.end_sec:.1f}s" for i, row in cat_df.reset_index(drop=True).iterrows()]
                    selected_event = st.selectbox(
                        "Inspect event",
                        event_opts,
                        key=f"blocker_evt_{category}",
                    )
                    sel_idx = int(selected_event.split(":", 1)[0])
                    sel_row = cat_df.reset_index(drop=True).iloc[sel_idx].to_dict()

                    if sel_row.get("operational_impact"):
                        st.markdown(f"**Operational impact:** {sel_row.get('operational_impact')}")
                    checks = sel_row.get("what_to_check_next")
                    if isinstance(checks, list) and checks:
                        st.markdown("**What to check next**")
                        for item in checks:
                            st.markdown(f"- {item}")

                    thumb_paths = sel_row.get("representative_frame_paths", [])
                    if isinstance(thumb_paths, str):
                        thumb_paths = [thumb_paths]
                    thumb_paths = [str(p) for p in thumb_paths[:3] if p]
                    if thumb_paths:
                        st.image(thumb_paths, width=140, caption=[Path(tp).name for tp in thumb_paths])

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
        st.dataframe(heat[["worker", "hour", "working_sec", "idle_sec", "transit_sec", "downtime_sec"]], height=200)

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
