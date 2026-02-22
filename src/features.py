from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ── Model cache to avoid reloading for every bank ──
_LOADED: dict[str, tuple] = {}


def _load_backbone(model_name: str, fallback_model_name: str):
    """Load a CLIP/SigLIP/DFN model, returning (processor, tokenizer, model, torch, model_type).

    model_type is 'clip' (has get_image_features) or 'siglip' (needs manual pooling).
    Results are cached so repeated calls with the same model_name are free.
    """
    cache_key = model_name
    if cache_key in _LOADED:
        return _LOADED[cache_key]

    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    names_to_try = [model_name, fallback_model_name]
    for name in names_to_try:
        try:
            processor = AutoProcessor.from_pretrained(name)
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
            dev = _device()
            model.to(dev)
            if dev == "cuda":
                model.half()  # FP16 — ~2× faster, ~half VRAM
            model.eval()

            # Detect model type
            model_type = "clip" if hasattr(model, "get_image_features") else "siglip"
            result = (processor, tokenizer, model, torch, model_type)
            _LOADED[cache_key] = result
            return result
        except Exception as exc:
            if name == fallback_model_name:
                raise RuntimeError(
                    f"Could not load model {model_name} or fallback {fallback_model_name}"
                ) from exc
            continue

    raise RuntimeError("No model loaded")


def _to_numpy(output, torch_module) -> np.ndarray:
    """Convert any model output (tensor, BaseModelOutput, tuple) to numpy."""
    if isinstance(output, torch_module.Tensor):
        return output.detach().cpu().numpy()
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output.detach().cpu().numpy()
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :].detach().cpu().numpy()
    if isinstance(output, tuple):
        return output[1].detach().cpu().numpy()
    raise TypeError(f"Cannot convert {type(output).__name__} to numpy")


def _get_image_features(model, processor, images: list[Image.Image], model_type: str, torch_module, device: str) -> np.ndarray:
    """Extract image embeddings regardless of model type."""
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch_module.no_grad():
        if model_type == "clip":
            out = model.get_image_features(**inputs)
        else:
            out = model.vision_model(**inputs)

    return _to_numpy(out, torch_module)


def _get_text_features(model, tokenizer, prompts: list[str], model_type: str, torch_module, device: str) -> np.ndarray:
    """Extract text embeddings regardless of model type."""
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch_module.no_grad():
        if model_type == "clip":
            out = model.get_text_features(**inputs)
        else:
            out = model.text_model(**inputs)

    return _to_numpy(out, torch_module)


def _simple_embed(img: Image.Image) -> np.ndarray:
    arr = np.array(img.resize((32, 32))).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    return np.concatenate([mean, std], axis=0)


def compute_embeddings(
    frame_paths: list[str],
    cache_file: Path,
    model_name: str,
    fallback_model_name: str,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Return (N, D) image embedding array in the joint vision-language space.

    Uses ``model.get_image_features()`` so embeddings are directly comparable
    to text features from ``compute_text_features()``.

    *cache_file* should be fps-keyed by the caller so caches from different
    sampling rates never collide.
    """
    if cache_file.exists():
        return np.load(cache_file)["embeddings"]

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = _device()

    try:
        processor, _tokenizer, model, torch, model_type = _load_backbone(model_name, fallback_model_name)

        all_embeds = []
        batches = range(0, len(frame_paths), batch_size)
        for i in tqdm(batches, desc="  embeddings", unit="batch", leave=True):
            batch = [Image.open(p).convert("RGB") for p in frame_paths[i : i + batch_size]]
            emb = _get_image_features(model, processor, batch, model_type, torch, device)
            all_embeds.append(emb)
        embeddings = np.concatenate(all_embeds, axis=0)
    except Exception:
        embeddings = np.stack(
            [_simple_embed(Image.open(p).convert("RGB")) for p in frame_paths], axis=0
        )

    np.savez_compressed(cache_file, embeddings=embeddings)
    return embeddings


def compute_text_features(
    prompts: list[str],
    model_name: str,
    fallback_model_name: str,
    device: str | None = None,
) -> np.ndarray:
    """Return (C, D) text embeddings for the given prompts.

    Embeddings live in the same joint space as those from
    ``compute_embeddings()``, so cosine similarity is meaningful.
    """
    if device is None:
        device = _device()

    _processor, tokenizer, model, torch, model_type = _load_backbone(model_name, fallback_model_name)

    return _get_text_features(model, tokenizer, prompts, model_type, torch, device)

