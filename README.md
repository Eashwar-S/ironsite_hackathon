## Egocentric Construction Productivity Analyzer

End-to-end pipeline to ingest egocentric construction videos, generate temporal state segments (IDLE / WORKING / TRANSIT), compute productivity analytics and rankings, and visualize results in Streamlit.

## Why this architecture

The system uses a staged architecture (frame-level signals + temporal smoothing + selective vision-language refinement) to balance **speed**, **stability**, and **semantic depth**:

1. **Fast iteration with caching:** expensive steps (frame extraction, embeddings, motion, refined VLM outputs) are cached, so reruns are incremental.
2. **Robust baseline behavior:** deterministic descriptor-bank scoring and smoothing produce stable coarse segments even without large-model calls.
3. **GPU budget where it matters:** refined labeling is selective/policy-driven instead of running on every frame, improving cost-performance.
4. **Operational modularity:** ingest, inference, analytics, and UI are separated, making it easier to tune FPS, change models, or disable refinement without rewrites.

## System diagram

![System architecture diagram](system_diagram.png)

The architecture is intentionally split into modular layers (ingest/cache → per-frame inference → temporal segmentation → analytics/reporting → Streamlit). This structure was chosen to keep each layer independently testable and replaceable while maintaining end-to-end reproducibility.

## Project Structure

- `src/` pipeline modules (ingestion, frames, embeddings, descriptor banks, motion, classification, smoothing, refined labels, blockers, analytics, reporting)
- `app/` Streamlit app + UI helpers
- `outputs/cache/` cached frames, embeddings, motion features, and VLM responses
- `outputs/runs/latest/` generated artifacts (`segments.json`, `metrics.json`, `rankings.json`, `daily_report.md`)

## Setup (Python 3.11+)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install CUDA-enabled PyTorch first (example: CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining project dependencies
pip install -r requirements.txt
cp .env.example .env
```

### GPU note
- Qwen2.5-VL (4-bit) and refined-label inference are intended for CUDA GPUs (target: RTX 3080 16GB or better).
- Verify CUDA availability: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
- Set `REFINED_VLM_4BIT=true` and ensure `bitsandbytes` is CUDA-compatible.

## Coarse vs refined labels

- **Coarse states** are predicted by CLIP/SigLIP + motion (`IDLE`, `WORKING`, `TRANSIT`, `DOWNTIME`) and drive core segmentation.
- **Refined labels** are an optional VLM layer (`WORKING`, `TRANSIT`, `IDLE_WAITING`, `IDLE_DEVICE`, `WORKING_PAUSE`, `UNSAFE_BEHAVIOR`, `BREAK`, `UNCERTAIN`).
- When enabled, refined predictions are merged into per-frame outputs and aggregated per segment as:
  - `dominant_refined_label`
  - `refined_top2`
  - `refined_notes`

## Integration plan / architecture

1. Segmentation pipeline remains unchanged through `to_segments()`.
2. Post-processing step (`src/blockers.py`) analyzes only long IDLE bursts and calls Qwen2.5-VL via `src/vlm.py` with on-disk/in-memory caching.
3. `metrics.json` is extended with `idle_blockers` and `idle_blocker_summary` while preserving existing fields.
4. `src/analytics.py` builds ranking score breakdowns and a time-of-day productivity heatmap dataset.
5. `src/report.py` generates daily markdown reports (OpenAI optional, deterministic fallback always available) and PDF bytes for export.
6. Streamlit dashboard exposes Overview, Rankings, Heatmap, and Daily Report tabs.

## Run pipeline

```bash
python -m src.pipeline --dataset dataset --out outputs/runs/latest --fps 1 --refined_labels true
```

## Launch UI

```bash
streamlit run app/streamlit_app.py
```

## Validation checklist

```bash
# Run a small subset manually by placing two videos in dataset/, then run pipeline:
python -m src.pipeline --dataset dataset --out outputs/runs/latest --fps 1 --refined_labels true
python scripts/validate_subset.py
```

## Troubleshooting

- **bitsandbytes/CUDA errors**: install a CUDA-compatible `bitsandbytes` build and verify `torch.cuda.is_available()`.
- **HF token missing**: set `HF_TOKEN` in `.env` for gated model access.
- **Out of memory**: reduce `BLOCKER_MAX_FRAMES`, lower extraction `FPS`, or set `REFINED_VLM_4BIT=true`.
- **No report LLM credentials**: set `REPORT_LLM_PROVIDER=none` to use deterministic report generation.

## Refined-label environment keys

```bash
REFINED_VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
REFINED_VLM_4BIT=true
REFINED_INFER_MODE=selective   # selective|all
REFINED_MAX_FRAMES_PER_VIDEO=1200
REFINED_CALL_POLICY=idle_and_pause   # uncertain_only|idle_and_pause|all
REFINED_BATCH_SIZE=4
REFINED_MAX_NEW_TOKENS=200
REFINED_TEMPERATURE=0.1
```

## Performance tips

- Use `REFINED_INFER_MODE=selective` with `REFINED_CALL_POLICY=idle_and_pause` for best speed/quality tradeoff.
- Refined outputs are cached per video under `outputs/cache/refined/<video_id>.jsonl` and reused across UI refreshes.
- `--refined_labels false` (default) skips the VLM stage entirely.
