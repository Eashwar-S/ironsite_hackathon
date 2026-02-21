## Egocentric Construction Productivity Analyzer

End-to-end pipeline to ingest egocentric construction videos, generate temporal state segments (IDLE / WORKING / TRANSIT), compute productivity analytics and rankings, and visualize results in Streamlit.

## Project Structure

- `src/` pipeline modules (ingestion, frames, embeddings, motion, classification, smoothing, blockers, VLM, reporting)
- `app/` Streamlit app + UI helpers
- `outputs/cache/` cached frames, embeddings, motion features, and VLM responses
- `outputs/runs/latest/` generated artifacts (`segments.json`, `metrics.json`, `rankings.json`, `daily_report.md`)

## Setup (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### GPU note
- Qwen2.5-VL (4-bit) is intended for CUDA GPUs (target: RTX 3080 16GB).
- Set `VLM_4BIT=true` and ensure `bitsandbytes` CUDA build is available.

## Integration plan / architecture

1. Segmentation pipeline remains unchanged through `to_segments()`.
2. Post-processing step (`src/blockers.py`) analyzes only long IDLE bursts and calls Qwen2.5-VL via `src/vlm.py` with on-disk/in-memory caching.
3. `metrics.json` is extended with `idle_blockers` and `idle_blocker_summary` while preserving existing fields.
4. `src/analytics.py` now builds ranking score breakdowns and a time-of-day productivity heatmap dataset.
5. `src/report.py` generates daily markdown reports (OpenAI optional, deterministic fallback always available) and PDF bytes for export.
6. Streamlit dashboard adds Overview, Rankings, Heatmap, and Daily Report tabs.

## Run pipeline

```bash
python -m src.pipeline --dataset dataset --out outputs/runs/latest --fps 1
```

## Launch UI

```bash
streamlit run app/streamlit_app.py
```

## Validation checklist

```bash
# Run a small subset manually by placing two videos in dataset/, then run pipeline:
python -m src.pipeline --dataset dataset --out outputs/runs/latest --fps 1
python scripts/validate_subset.py
```

## Troubleshooting

- **bitsandbytes/CUDA errors**: install a CUDA-compatible `bitsandbytes` build, verify `torch.cuda.is_available()`.
- **HF token missing**: set `HF_TOKEN` in `.env` for gated model access.
- **Out of memory**: reduce `BLOCKER_MAX_FRAMES`, lower extraction `FPS`, or set `VLM_4BIT=true`.
- **No report LLM credentials**: set `REPORT_LLM_PROVIDER=none` to use deterministic report generation.
