from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Config:
    dataset_dir: Path = Path(os.getenv("DATASET_DIR", "dataset"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    run_id: str = os.getenv("RUN_ID", "latest")

    fps: float = float(os.getenv("FPS", "1.0"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "0"))

    model_name: str = os.getenv("MODEL_NAME", "google/siglip-base-patch16-224")
    fallback_model_name: str = os.getenv("FALLBACK_MODEL_NAME", "openai/clip-vit-base-patch32")

    clip_logit_scale: float = float(os.getenv("CLIP_LOGIT_SCALE", "100.0"))
    smoothing_window: int = int(os.getenv("SMOOTHING_WINDOW", "9"))
    idle_burst_seconds: float = float(os.getenv("IDLE_BURST_SECONDS", "8"))

    w_working: float = float(os.getenv("W_WORKING", "1.0"))
    w_idle: float = float(os.getenv("W_IDLE", "0.8"))
    w_transit: float = float(os.getenv("W_TRANSIT", "0.5"))
    w_transitions: float = float(os.getenv("W_TRANSITIONS", "0.2"))
    w_idle_bursts: float = float(os.getenv("W_IDLE_BURSTS", "0.15"))

    hf_token: str = os.getenv("HF_TOKEN", "")
    vlm_model: str = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    vlm_device: str = os.getenv("VLM_DEVICE", "cuda")
    vlm_4bit: bool = os.getenv("VLM_4BIT", "false").lower() in {"1", "true", "yes"}
    vlm_max_new_tokens: int = int(os.getenv("VLM_MAX_NEW_TOKENS", "256"))
    vlm_temperature: float = float(os.getenv("VLM_TEMPERATURE", "0.2"))

    idle_burst_sec: float = float(os.getenv("IDLE_BURST_SEC", "20"))
    blocker_frame_sample_fps: float = float(os.getenv("BLOCKER_FRAME_SAMPLE_FPS", "1"))
    blocker_max_frames: int = int(os.getenv("BLOCKER_MAX_FRAMES", "12"))

    report_llm_provider: str = os.getenv("REPORT_LLM_PROVIDER", "none")
    report_model: str = os.getenv("REPORT_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    day_start_time: str = os.getenv("DAY_START_TIME", "07:00")

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.output_dir / "runs"

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id


CONFIG = Config()
