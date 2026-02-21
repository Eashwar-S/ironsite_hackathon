from __future__ import annotations

import json

import pandas as pd
import plotly.express as px

COLOR_MAP   = {"WORKING": "#2ca02c", "IDLE": "#d62728", "TRANSIT": "#1f77b4"}
LABEL_ORDER = ["IDLE", "TRANSIT", "WORKING"]


def canvas_timeline_html(segments: list[dict], current_t: float = 0.0) -> str:
    """Self-contained HTML canvas timeline bar.

    When the user clicks the bar, it sends the clicked time (seconds) back
    to Streamlit via query-param manipulation so the slider can pick it up.
    A white playhead line is drawn at *current_t*.
    Hovering shows a tooltip with the segment label and time range.
    """
    segs_json   = json.dumps(segments)
    colors_json = json.dumps(COLOR_MAP)
    max_t = max((s["end_sec"] for s in segments), default=60.0)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0e1117;
    color: #eee;
    font-family: sans-serif;
    padding: 6px 8px 0;
    overflow: hidden;
  }}
  #bar {{
    position: relative;
    width: 100%;
    height: 44px;
    cursor: pointer;
    border-radius: 6px;
    overflow: hidden;
  }}
  #tl-canvas {{ display: block; width: 100%; height: 100%; }}
  #meta {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 4px;
    font-size: 11px;
    color: #aaa;
  }}
  .legend {{ display: flex; gap: 14px; }}
  .leg {{ display: flex; align-items: center; gap: 4px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
  #tooltip {{
    position: absolute;
    background: rgba(0,0,0,0.85);
    color: #fff;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    pointer-events: none;
    display: none;
    white-space: nowrap;
    z-index: 10;
  }}
</style>
</head>
<body>
<div id="bar"><canvas id="tl-canvas"></canvas></div>
<div id="tooltip"></div>
<div id="meta">
  <span id="tl-info">Click the bar to jump to a time</span>
  <div class="legend">
    <div class="leg"><div class="dot" style="background:#d62728"></div>IDLE</div>
    <div class="leg"><div class="dot" style="background:#1f77b4"></div>TRANSIT</div>
    <div class="leg"><div class="dot" style="background:#2ca02c"></div>WORKING</div>
  </div>
</div>

<script>
const segments  = {segs_json};
const colors    = {colors_json};
const maxT      = {max_t};
const currentT  = {current_t};
const canvas    = document.getElementById('tl-canvas');
const ctx       = canvas.getContext('2d');
const tooltip   = document.getElementById('tooltip');
const infoEl    = document.getElementById('tl-info');

function resize() {{
  const r = canvas.getBoundingClientRect();
  canvas.width  = Math.round(r.width  * devicePixelRatio);
  canvas.height = Math.round(r.height * devicePixelRatio);
}}

function segAt(t) {{
  for (const s of segments)
    if (t >= s.start_sec && t < s.end_sec) return s;
  return null;
}}

function draw(t) {{
  const W = canvas.width  / devicePixelRatio;
  const H = canvas.height / devicePixelRatio;
  ctx.save();
  ctx.scale(devicePixelRatio, devicePixelRatio);

  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);

  for (const s of segments) {{
    const x0 = (s.start_sec / maxT) * W;
    const x1 = (s.end_sec   / maxT) * W;
    ctx.fillStyle = colors[s.label] || '#555';
    ctx.fillRect(x0, 2, Math.max(1, x1 - x0), H - 4);
  }}

  // Playhead
  const xNow = Math.round((t / maxT) * W);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth   = 2.5;
  ctx.shadowColor = '#fff';
  ctx.shadowBlur  = 8;
  ctx.beginPath();
  ctx.moveTo(xNow, 0);
  ctx.lineTo(xNow, H);
  ctx.stroke();

  ctx.restore();
}}

// Hover tooltip
canvas.addEventListener('mousemove', e => {{
  const r = canvas.getBoundingClientRect();
  const t = ((e.clientX - r.left) / r.width) * maxT;
  const s = segAt(t);
  if (s) {{
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX - r.left + 12) + 'px';
    tooltip.style.top  = '-20px';
    tooltip.textContent = s.label + '  ' + s.start_sec.toFixed(1) + 's – ' + s.end_sec.toFixed(1) + 's';
    tooltip.style.borderLeft = '3px solid ' + (colors[s.label] || '#fff');
  }} else {{
    tooltip.style.display = 'none';
  }}
}});
canvas.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

// Click → send time to Streamlit via query param
document.getElementById('bar').addEventListener('click', e => {{
  const r = canvas.getBoundingClientRect();
  const t = Math.round(((e.clientX - r.left) / r.width) * maxT);
  const url = new URL(window.parent.location.href);
  url.searchParams.set('tl_click', t);
  window.parent.history.replaceState(null, '', url);
  window.parent.postMessage({{type: 'timeline_click', time: t}}, '*');
  infoEl.textContent = '⏩ Clicked: ' + t + 's';
  draw(t);
}});

window.addEventListener('resize', () => {{ resize(); draw(currentT); }});
resize();
draw(currentT);
</script>
</body>
</html>"""


def metrics_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Metric": ["Working %", "Idle %", "Transit %", "Transitions/min", "Idle burst count"],
        "Value": [
            f"{metrics.get('working_pct', 0) * 100:.1f}%",
            f"{metrics.get('idle_pct', 0) * 100:.1f}%",
            f"{metrics.get('transit_pct', 0) * 100:.1f}%",
            f"{metrics.get('transitions_per_min', 0):.2f}",
            str(int(metrics.get("idle_burst_count", 0))),
        ],
    })


def blocker_chart(summary: dict):
    data = summary.get("counts_by_category", {}) if summary else {}
    if not data:
        return None
    df = pd.DataFrame({"category": list(data.keys()), "count": list(data.values())})
    fig = px.bar(df, x="category", y="count", title="Idle blockers by category")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig
