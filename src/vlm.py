from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .config import CONFIG


@dataclass
class VLMRuntime:
    model: Any
    processor: Any


_VLM_RUNTIME: VLMRuntime | None = None
_VLM_ERROR: str | None = None
_MEMORY_CACHE: dict[str, str] = {}


def _cache_file() -> Path:
    cache_dir = CONFIG.cache_dir / "vlm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "cache.json"


def _load_disk_cache() -> dict[str, str]:
    cache_file = _cache_file()
    if not cache_file.exists():
        return {}
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_disk_cache(payload: dict[str, str]) -> None:
    _cache_file().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _hash_payload(images: list[Image.Image], prompt: str) -> str:
    h = hashlib.sha256(prompt.encode("utf-8"))
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        h.update(hashlib.sha256(buf.getvalue()).digest())
    return h.hexdigest()


def load_vlm() -> VLMRuntime:
    global _VLM_RUNTIME, _VLM_ERROR
    if _VLM_RUNTIME is not None:
        return _VLM_RUNTIME
    if _VLM_ERROR:
        raise RuntimeError(_VLM_ERROR)

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except Exception as exc:
        _VLM_ERROR = f"Missing VLM dependencies (transformers/torch/bitsandbytes): {exc}"
        raise RuntimeError(_VLM_ERROR) from exc

    if CONFIG.vlm_device.startswith("cuda") and not torch.cuda.is_available():
        _VLM_ERROR = "VLM_DEVICE=cuda but CUDA is not available."
        raise RuntimeError(_VLM_ERROR)

    model_kwargs: dict[str, Any] = {}
    if CONFIG.vlm_4bit:
        try:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["device_map"] = "auto"
        except Exception as exc:
            _VLM_ERROR = f"Unable to initialize 4-bit quantization. Install bitsandbytes with CUDA support: {exc}"
            raise RuntimeError(_VLM_ERROR) from exc
    else:
        model_kwargs["torch_dtype"] = torch.float16

    token = CONFIG.hf_token or None
    processor = AutoProcessor.from_pretrained(CONFIG.vlm_model, token=token, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        CONFIG.vlm_model,
        token=token,
        trust_remote_code=True,
        **model_kwargs,
    )
    if not CONFIG.vlm_4bit:
        model = model.to(CONFIG.vlm_device)
    _VLM_RUNTIME = VLMRuntime(model=model, processor=processor)
    return _VLM_RUNTIME


def vlm_infer(images: list[Image.Image], prompt: str) -> str:
    if not images:
        raise ValueError("vlm_infer requires at least one image.")

    cache_key = _hash_payload(images, prompt)
    if cache_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_key]

    disk_cache = _load_disk_cache()
    if cache_key in disk_cache:
        _MEMORY_CACHE[cache_key] = disk_cache[cache_key]
        return disk_cache[cache_key]

    runtime = load_vlm()
    processor = runtime.processor
    model = runtime.model

    chat = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}],
        }
    ]
    text_prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=CONFIG.vlm_max_new_tokens,
        temperature=CONFIG.vlm_temperature,
        do_sample=CONFIG.vlm_temperature > 0,
    )
    generated = output_ids[:, inputs["input_ids"].shape[-1] :]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    _MEMORY_CACHE[cache_key] = text
    disk_cache[cache_key] = text
    _save_disk_cache(disk_cache)
    return text
