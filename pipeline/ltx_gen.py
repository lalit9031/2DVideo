"""
pipeline/ltx_gen.py
~~~~~~~~~~~~~~~~~~~~
Local LTX-2.x B-roll video generation via HuggingFace diffusers.

Uses LTXPipeline (or LTXImageToVideoPipeline) from the diffusers library,
running on the system ROCm/CUDA GPU.  Falls back silently to the procedural
render_broll() if the model is unavailable or VRAM is insufficient.
"""
from __future__ import annotations

import gc
import os
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline.common import ensure_dir

# ---------------------------------------------------------------------------
# Model cache — loaded once, reused across shots in the same process
# ---------------------------------------------------------------------------
_LTX_PIPE: object | None = None
_LTX_LOAD_FAILED: bool = False

# Resolution for LTX generation — keep divisible by 32 and ≤ 768 to stay in VRAM
_LTX_W = 704
_LTX_H = 480
_LTX_FRAMES = 25        # ≈1 s at 25 fps; will be trimmed/looped to shot duration
_LTX_FPS   = 25


def _load_ltx() -> object | None:
    global _LTX_PIPE, _LTX_LOAD_FAILED
    if _LTX_LOAD_FAILED:
        return None
    if _LTX_PIPE is not None:
        return _LTX_PIPE
    try:
        import torch
        from diffusers import LTXPipeline

        # Use ROCm/CUDA if available, else CPU (very slow but doesn't crash)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.bfloat16 if device == "cuda" else torch.float32

        # Model id — use the smaller distilled variant if available, else base
        model_id = os.environ.get(
            "LTX_MODEL_ID",
            "Lightricks/LTX-Video",
        )
        pipe = LTXPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        # Memory optimisation for 24 GB VRAM: enable attention slicing
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        _LTX_PIPE = pipe
        return _LTX_PIPE
    except Exception as exc:
        warnings.warn(
            f"LTX model could not be loaded: {exc}. "
            "B-roll will use the procedural fallback renderer."
        )
        _LTX_LOAD_FAILED = True
        return None


def _frames_to_uint8(tensor) -> list[np.ndarray]:
    """Convert a diffusers video tensor (B, C, T, H, W) or (T, H, W, C) to list of RGB uint8 arrays."""
    import torch

    # diffusers LTXPipeline returns frames attribute on the output object
    if hasattr(tensor, "frames"):
        frames_list = tensor.frames[0]   # list of PIL images or np arrays
        out = []
        for f in frames_list:
            if hasattr(f, "numpy"):       # PIL image
                arr = np.array(f)
            else:
                arr = np.asarray(f)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            out.append(arr)
        return out

    # Fallback: tensor shape (T, H, W, C) float [0,1]
    t = tensor.cpu().float().numpy()
    if t.ndim == 4 and t.shape[-1] in (1, 3, 4):
        return [(np.clip(t[i], 0, 1) * 255).astype(np.uint8) for i in range(t.shape[0])]
    return []


def generate_broll_ltx(
    prompt: str,
    output_path: Path,
    *,
    duration_sec: float = 4.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 24.0,
    negative_prompt: str = "low quality, blur, watermark, text, ugly, distorted",
    num_inference_steps: int = 30,
    guidance_scale: float = 3.5,
    progress_callback: Callable[[float], None] | None = None,
) -> bool:
    """
    Generate a B-roll clip with LTX.
    Returns True if LTX was used, False if the fallback should be used.
    """
    pipe = _load_ltx()
    if pipe is None:
        return False

    try:
        import torch
        from pipeline.media import export_video

        ensure_dir(output_path.parent)

        # Generate at LTX native resolution, then resize to target
        num_frames = max(_LTX_FRAMES, int(round(duration_sec * _LTX_FPS)))
        # LTX requires num_frames to satisfy (num_frames - 1) % 8 == 0
        num_frames = ((num_frames - 1) // 8) * 8 + 1

        generator = torch.manual_seed(abs(hash(prompt)) % (2 ** 31))

        if progress_callback:
            progress_callback(0.05)

        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=_LTX_W,
                height=_LTX_H,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        if progress_callback:
            progress_callback(0.85)

        raw_frames = _frames_to_uint8(output)
        if not raw_frames:
            return False

        # Resize to target output resolution
        import cv2
        target_frames_count = max(1, int(round(duration_sec * fps)))
        resized: list[np.ndarray] = []
        for arr in raw_frames:
            resized.append(cv2.resize(arr, (width, height), interpolation=cv2.INTER_LANCZOS4))

        # Loop or trim to exact target frame count
        if len(resized) < target_frames_count:
            factor = -(-target_frames_count // len(resized))   # ceil division
            resized = (resized * factor)[:target_frames_count]
        else:
            resized = resized[:target_frames_count]

        export_video(resized, output_path, fps)

        if progress_callback:
            progress_callback(1.0)

        # Free VRAM after generation — we won't need it for a while
        del output, raw_frames, resized
        gc.collect()
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()

        return True

    except Exception as exc:
        warnings.warn(f"LTX generation failed ({exc}); using procedural fallback.")
        return False
