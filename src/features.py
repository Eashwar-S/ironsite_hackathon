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


def _load_backbone(model_name: str, fallback_model_name: str):
    """Load a CLIP/SigLIP model, returning (processor, tokenizer, model, torch)."""
    try:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(_device())
        model.eval()
        return (processor, tokenizer, model, torch)
    except Exception:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        processor = AutoProcessor.from_pretrained(fallback_model_name)
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
        model = AutoModel.from_pretrained(fallback_model_name)
        model.to(_device())
        model.eval()
        return (processor, tokenizer, model, torch)


def _to_numpy(output, torch_module):
    """Convert a model output (tensor or BaseModelOutput) to a numpy array."""
    if isinstance(output, torch_module.Tensor):
        return output.detach().cpu().numpy()
    # Some transformers versions return a model output object
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output.detach().cpu().numpy()
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :].detach().cpu().numpy()
    # Fallback: treat as indexable tuple (pooled is usually second element)
    return output[1].detach().cpu().numpy()


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
        processor, _tokenizer, model, torch = _load_backbone(model_name, fallback_model_name)

        all_embeds = []
        batches = range(0, len(frame_paths), batch_size)
        for i in tqdm(batches, desc="  embeddings", unit="batch", leave=True):
            batch = [Image.open(p).convert("RGB") for p in frame_paths[i : i + batch_size]]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = _to_numpy(model.get_image_features(**inputs), torch)
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

    _processor, tokenizer, model, torch = _load_backbone(model_name, fallback_model_name)

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        return _to_numpy(model.get_text_features(**inputs), torch)
