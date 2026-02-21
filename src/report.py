from __future__ import annotations

from collections import Counter
from io import BytesIO


def _rule_based_report(metrics: list[dict], rankings: dict) -> str:
    lines = ["# Daily Productivity Report", "", "## Executive summary"]
    if not metrics:
        return "# Daily Productivity Report\n\nNo metrics available."

    avg_working = sum(m.get("working_pct", 0) for m in metrics) / len(metrics)
    lines.append(f"- Average working share across workers: **{avg_working * 100:.1f}%**.")

    blocker_counter = Counter()
    for m in metrics:
        for cat, count in m.get("idle_blocker_summary", {}).get("counts_by_category", {}).items():
            blocker_counter[cat] += count
    if blocker_counter:
        lines.append(f"- Top blocker category: **{blocker_counter.most_common(1)[0][0]}**.")

    lines.extend(["", "## Rankings"])
    for task, rows in rankings.items():
        lines.append(f"### {task}")
        for idx, row in enumerate(rows[:5], start=1):
            lines.append(
                f"{idx}. {row['person_id']} ({row['video_id']}): score {row['score']:.3f}, "
                f"working {row['working_pct']*100:.1f}%, idle {row['idle_pct']*100:.1f}%"
            )

    lines.extend(["", "## Recommendations"])
    if blocker_counter:
        for cat, _ in blocker_counter.most_common(3):
            lines.append(f"- Reduce '{cat}' with targeted pre-shift planning and supervisor checkpoints.")
    lines.append("- Investigate workers with high idle burst counts and long transit windows.")
    lines.append("- Verify material/tool staging readiness at shift start.")
    return "\n".join(lines)


def generate_daily_report(metrics: list[dict], rankings: dict, provider: str = "none", model: str = "") -> str:
    provider = (provider or "none").lower()
    if provider != "openai":
        return _rule_based_report(metrics, rankings)

    try:
        from openai import OpenAI
    except Exception:
        return _rule_based_report(metrics, rankings)

    prompt = (
        "Create a concise construction productivity daily report in markdown with sections: "
        "Executive summary, Rankings, Worker-by-worker notes, Top blockers, Recommendations, Next steps.\n"
        f"Metrics: {metrics}\nRankings: {rankings}"
    )
    client = OpenAI()
    try:
        resp = client.responses.create(model=model or "gpt-4o-mini", input=prompt)
        return resp.output_text
    except Exception:
        return _rule_based_report(metrics, rankings)


def report_to_pdf(markdown: str) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in markdown.splitlines():
        text = line.replace("#", "").strip()
        if not text:
            y -= 10
        else:
            pdf.drawString(40, y, text[:130])
            y -= 14
        if y < 40:
            pdf.showPage()
            y = height - 40
    pdf.save()
    return buffer.getvalue()
