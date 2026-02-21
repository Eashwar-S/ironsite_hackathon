# vlm.py
from __future__ import annotations
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image
from .config import CONFIG
from .hypothesis import LABELS

# ============================================================================
# VLM Runtime & Core Inference
# ============================================================================

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
    """Core VLM inference with caching."""
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

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Segment:
    """Consecutive frames with the same label."""
    label: str
    start_frame: int
    end_frame: int
    frame_indices: list[int]
    duration_frames: int
    vlm_analysis: str | None = None
    vlm_structured: dict | None = None

# ============================================================================
# Segmentation Logic
# ============================================================================

def segment_labels(frame_labels: list[str]) -> list[Segment]:
    """
    Group consecutive frames with same label into segments.
    
    Args:
        frame_labels: List of label strings, one per frame
    
    Returns:
        List of Segment objects
    """
    if not frame_labels:
        return []
    
    segments = []
    current_label = frame_labels[0]
    start_frame = 0
    frame_indices = [0]
    
    for i, label in enumerate(frame_labels[1:], start=1):
        if label == current_label:
            frame_indices.append(i)
        else:
            # Save current segment
            segments.append(Segment(
                label=current_label,
                start_frame=start_frame,
                end_frame=i - 1,
                frame_indices=frame_indices,
                duration_frames=len(frame_indices),
            ))
            # Start new segment
            current_label = label
            start_frame = i
            frame_indices = [i]
    
    # Add final segment
    segments.append(Segment(
        label=current_label,
        start_frame=start_frame,
        end_frame=len(frame_labels) - 1,
        frame_indices=frame_indices,
        duration_frames=len(frame_indices),
    ))
    
    return segments

def should_analyze_with_vlm(segment: Segment, min_duration_frames: int = 10) -> bool:
    """
    Determine if segment needs VLM analysis.
    
    Args:
        segment: Segment to evaluate
        min_duration_frames: Minimum segment length to process (default 10)
    
    Returns:
        True if VLM analysis is needed
    """
    # Check if this label has a VLM hypothesis
    if segment.label not in LABELS:
        return False
    
    if LABELS[segment.label].get("vlm_hypothesis") is None:
        return False
    
    # Skip very short segments to save compute
    if segment.duration_frames < min_duration_frames:
        return False
    
    return True

def sample_segment_frames(
    segment: Segment, 
    all_frames: list[Image.Image], 
    max_frames: int = 3
) -> list[Image.Image]:
    """
    Sample key frames from a segment for VLM analysis.
    
    Args:
        segment: Segment to sample from
        all_frames: Full list of frame images
        max_frames: Maximum number of frames to sample
    
    Returns:
        List of sampled PIL Images
    """
    indices = segment.frame_indices
    n_frames = len(indices)
    
    if n_frames <= max_frames:
        # Return all frames if segment is short
        return [all_frames[i] for i in indices]
    
    # Sample strategically based on max_frames
    if max_frames == 1:
        sample_indices = [indices[n_frames // 2]]  # Middle frame
    elif max_frames == 2:
        sample_indices = [indices[0], indices[-1]]  # Start and end
    elif max_frames == 3:
        sample_indices = [indices[0], indices[n_frames // 2], indices[-1]]
    elif max_frames == 5:
        sample_indices = [
            indices[0],
            indices[n_frames // 4],
            indices[n_frames // 2],
            indices[3 * n_frames // 4],
            indices[-1]
        ]
    else:
        # Uniform sampling
        step = n_frames / max_frames
        sample_indices = [indices[int(i * step)] for i in range(max_frames)]
    
    return [all_frames[i] for i in sample_indices]

# ============================================================================
# Prompt Building
# ============================================================================

def build_vlm_prompt(segment: Segment, fps: float = 30.0) -> str:
    """
    Build VLM prompt based on segment label and hypothesis.
    
    Args:
        segment: Segment to analyze
        fps: Frames per second for duration calculation
    
    Returns:
        Prompt string for VLM
    """
    duration_sec = segment.duration_frames / fps
    
    if segment.label == "WORKING_PAUSE":
        return (
            f"These images show {segment.duration_frames} frames ({duration_sec:.1f} seconds) "
            f"of a construction worker during a work pause.\n\n"
            f"Analyze the sequence and classify as ONE of these sub-types:\n"
            f"1. TOOL_ISSUE - worker holding tool but not using it (possible malfunction or wrong tool)\n"
            f"2. INSPECTION_PAUSE - worker checking, measuring, or inspecting completed work\n"
            f"3. MATERIAL_SEARCH - worker with empty hands looking around for materials or tools\n\n"
            f"Respond with the sub-type name in ALL CAPS, followed by a colon and a brief 1-sentence explanation.\n"
            f"Example: 'TOOL_ISSUE: Worker holding drill but not applying it to surface, may be malfunctioning.'"
        )
    
    elif segment.label == "IDLE_WAITING":
        return (
            f"These images show {segment.duration_frames} frames ({duration_sec:.1f} seconds) "
            f"of a construction worker who appears idle with no active task.\n\n"
            f"What is the most likely reason for this idle time? Consider:\n"
            f"- Waiting for material delivery\n"
            f"- Waiting for another trade/worker to finish\n"
            f"- Unclear task assignment or direction\n"
            f"- Waiting for supervision or instruction\n\n"
            f"Provide a brief 1-2 sentence assessment of the most likely cause."
        )
    
    elif segment.label == "IDLE_DEVICE":
        return (
            f"These images show {segment.duration_frames} frames ({duration_sec:.1f} seconds) "
            f"of a worker using a phone or device.\n\n"
            f"Assess whether this appears to be:\n"
            f"A) PERSONAL use (social media, personal calls, browsing)\n"
            f"B) WORK-RELATED use (calling supervisor, checking plans/specs, work coordination)\n\n"
            f"Start your response with 'PERSONAL' or 'WORK-RELATED', followed by a colon and "
            f"a brief explanation of your reasoning based on visible context."
        )
    
    elif segment.label == "UNSAFE_BEHAVIOR":
        return (
            f"These images show {segment.duration_frames} frames ({duration_sec:.1f} seconds) "
            f"flagged for potential safety concerns.\n\n"
            f"Identify any visible safety issues such as:\n"
            f"- Missing or inadequate PPE (gloves, hard hat, safety glasses, etc.)\n"
            f"- Awkward or unsafe body positioning\n"
            f"- Working at elevation without proper fall protection\n"
            f"- Hands in enclosed or hazardous spaces without protection\n"
            f"- Other safety violations\n\n"
            f"If you identify safety concerns, list them clearly. If this is a false positive "
            f"with no actual safety issues, respond with 'FALSE_POSITIVE' and explain why."
        )
    
    # Fallback to hypothesis text if available
    hypothesis = LABELS.get(segment.label, {}).get("vlm_hypothesis")
    if hypothesis:
        return (
            f"These images show {segment.duration_frames} frames ({duration_sec:.1f} seconds) "
            f"of video footage.\n\n{hypothesis}"
        )
    
    return f"Analyze these {segment.duration_frames} frames and describe what the worker is doing."

# ============================================================================
# Response Parsing
# ============================================================================

def parse_vlm_response(segment: Segment, vlm_text: str) -> dict:
    """
    Parse VLM response into structured data.
    
    Args:
        segment: Segment that was analyzed
        vlm_text: Raw VLM response text
    
    Returns:
        Dictionary with structured results
    """
    result = {
        "raw_response": vlm_text,
        "label": segment.label,
    }
    
    if segment.label == "WORKING_PAUSE":
        # Try to extract subtype
        upper_text = vlm_text.upper()
        if "TOOL_ISSUE" in upper_text:
            result["subtype"] = "TOOL_ISSUE"
        elif "INSPECTION_PAUSE" in upper_text:
            result["subtype"] = "INSPECTION_PAUSE"
        elif "MATERIAL_SEARCH" in upper_text:
            result["subtype"] = "MATERIAL_SEARCH"
        else:
            result["subtype"] = "UNKNOWN"
        
        # Extract explanation (text after colon)
        if ":" in vlm_text:
            result["explanation"] = vlm_text.split(":", 1)[1].strip()
        else:
            result["explanation"] = vlm_text
    
    elif segment.label == "IDLE_DEVICE":
        upper_text = vlm_text.upper()
        if "PERSONAL" in upper_text and "WORK" not in upper_text[:20]:
            result["usage_type"] = "PERSONAL"
        elif "WORK" in upper_text[:20]:
            result["usage_type"] = "WORK-RELATED"
        else:
            result["usage_type"] = "UNCLEAR"
        
        if ":" in vlm_text:
            result["reasoning"] = vlm_text.split(":", 1)[1].strip()
        else:
            result["reasoning"] = vlm_text
    
    elif segment.label == "UNSAFE_BEHAVIOR":
        upper_text = vlm_text.upper()
        if "FALSE" in upper_text and "POSITIVE" in upper_text:
            result["is_unsafe"] = False
            result["safety_issues"] = []
        else:
            result["is_unsafe"] = True
            # Try to extract list of issues
            issues = []
            for line in vlm_text.split("\n"):
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                    issues.append(line.lstrip("-•0123456789. "))
            result["safety_issues"] = issues if issues else [vlm_text]
    
    elif segment.label == "IDLE_WAITING":
        result["idle_reason"] = vlm_text
    
    return result

# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_segments(
    frame_labels: list[str],
    frame_images: list[Image.Image],
    fps: float = 30.0,
    min_duration_frames: int = 10,
    max_frames_per_segment: int = 3,
) -> list[Segment]:
    """
    Full pipeline: segment labels, sample frames, run VLM analysis.
    
    Args:
        frame_labels: List of label strings, one per frame
        frame_images: List of PIL Images, one per frame
        fps: Frames per second for time calculations
        min_duration_frames: Minimum segment length to analyze with VLM
        max_frames_per_segment: Maximum frames to send to VLM per segment
    
    Returns:
        List of Segment objects with VLM analysis added where applicable
    """
    if len(frame_labels) != len(frame_images):
        raise ValueError("frame_labels and frame_images must have same length")
    
    # Step 1: Segment consecutive labels
    segments = segment_labels(frame_labels)
    
    # Step 2: Analyze segments that need VLM
    for segment in segments:
        if should_analyze_with_vlm(segment, min_duration_frames):
            # Sample frames
            sampled_images = sample_segment_frames(
                segment, 
                frame_images, 
                max_frames=max_frames_per_segment
            )
            
            # Build prompt
            prompt = build_vlm_prompt(segment, fps)
            
            # Run VLM inference (with caching)
            try:
                vlm_text = vlm_infer(sampled_images, prompt)
                segment.vlm_analysis = vlm_text
                
                # Parse into structured format
                segment.vlm_structured = parse_vlm_response(segment, vlm_text)
            except Exception as e:
                segment.vlm_analysis = f"ERROR: {str(e)}"
                segment.vlm_structured = {"error": str(e)}
    
    return segments

# ============================================================================
# Utility Functions
# ============================================================================

def get_segments_summary(segments: list[Segment], fps: float = 30.0) -> dict:
    """
    Generate summary statistics from analyzed segments.
    
    Args:
        segments: List of analyzed segments
        fps: Frames per second
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_segments": len(segments),
        "total_frames": sum(s.duration_frames for s in segments),
        "label_counts": {},
        "vlm_analyzed_count": 0,
        "total_duration_seconds": sum(s.duration_frames for s in segments) / fps,
    }
    
    for segment in segments:
        label = segment.label
        summary["label_counts"][label] = summary["label_counts"].get(label, 0) + 1
        
        if segment.vlm_analysis:
            summary["vlm_analyzed_count"] += 1
    
    return summary

def export_segments_json(segments: list[Segment], output_path: Path, fps: float = 30.0) -> None:
    """
    Export segments to JSON file.
    
    Args:
        segments: List of segments to export
        output_path: Path to output JSON file
        fps: Frames per second
    """
    data = {
        "summary": get_segments_summary(segments, fps),
        "segments": [
            {
                "label": s.label,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "duration_frames": s.duration_frames,
                "duration_seconds": s.duration_frames / fps,
                "vlm_analysis": s.vlm_analysis,
                "vlm_structured": s.vlm_structured,
            }
            for s in segments
        ]
    }
    
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    #hi