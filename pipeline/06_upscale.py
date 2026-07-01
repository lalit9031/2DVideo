"""
pipeline/06_upscale.py
~~~~~~~~~~~~~~~~~~~~~~
NEW: Upscale scene PNGs (not video) using Real-ESRGAN 2× before AnimateDiff.

Resolution targets:
  Short (9:16):  576×1024  → 1152×2048  (~2K portrait)
  Full  (16:9): 1024×576   → 2048×1152  (2K landscape)

Fallback chain:
  1. basicsr + realesrgan Python package
  2. realesrgan-ncnn-vulkan binary
  3. cv2.resize LANCZOS4

Also retains upscale_video_smart() for backward compatibility.
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ensure_dir, read_json, write_json
from pipeline.progress import progress_path_from_env, write_stage_progress

import numpy as np


# ---------------------------------------------------------------------------
# Real-ESRGAN Python API (upscale a single image array)
# ---------------------------------------------------------------------------

_ESRGAN_MODEL: object | None = None
_ESRGAN_FAILED: bool = False


def _get_esrgan(scale: int = 2):
    global _ESRGAN_MODEL, _ESRGAN_FAILED
    if _ESRGAN_FAILED:
        return None
    if _ESRGAN_MODEL is not None:
        return _ESRGAN_MODEL
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale,
        )
        # anime model works best for cartoon content
        model_name = "RealESRGAN_x2plus" if scale == 2 else "RealESRGAN_x4plus_anime_6B"
        _ESRGAN_MODEL = RealESRGANer(
            scale=scale, model_path=None, model=model,
            model_name=model_name,
            tile=512, tile_pad=10, pre_pad=0,
            half=True,
        )
        print(f"[upscale] Real-ESRGAN (Python) loaded, scale={scale}", flush=True)
        return _ESRGAN_MODEL
    except Exception as exc:
        warnings.warn(f"Real-ESRGAN Python API unavailable: {exc}")
        _ESRGAN_FAILED = True
        return None


def _upscale_png_esrgan(img_np: np.ndarray, scale: int = 2) -> np.ndarray | None:
    """Upscale an RGB numpy image. Returns upscaled RGB ndarray or None."""
    import cv2
    upsampler = _get_esrgan(scale)
    if upsampler is None:
        return None
    try:
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        enhanced, _ = upsampler.enhance(bgr, outscale=scale)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        warnings.warn(f"Real-ESRGAN enhance failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# ncnn-vulkan binary fallback (single image)
# ---------------------------------------------------------------------------

def _upscale_png_ncnn(input_png: Path, output_png: Path, scale: int = 2) -> bool:
    binary = shutil.which("realesrgan-ncnn-vulkan")
    if not binary:
        return False
    try:
        model_name = "realesrgan-x2plus-anime" if scale == 2 else "realesrgan-x4plus"
        cmd = [
            binary,
            "-i", str(input_png),
            "-o", str(output_png),
            "-n", model_name,
            "-s", str(scale),
            "-f", "png",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_png.exists()
    except Exception as exc:
        warnings.warn(f"ncnn-vulkan single-image failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# cv2 fallback
# ---------------------------------------------------------------------------

def _upscale_png_cv2(img_np: np.ndarray, scale: float) -> np.ndarray:
    import cv2
    h, w = img_np.shape[:2]
    return cv2.resize(img_np, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# Main PNG upscaler
# ---------------------------------------------------------------------------

def upscale_png(input_png: Path, output_png: Path, scale: int = 2) -> dict:
    """
    Upscale a single PNG file 2× using Real-ESRGAN → ncnn → cv2.
    Returns a report dict.
    """
    import cv2

    ensure_dir(output_png.parent)
    bgr = cv2.imread(str(input_png))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read {input_png}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]

    # Try ESRGAN Python
    upscaled = _upscale_png_esrgan(img, scale=scale)
    method = "realesrgan-python"

    if upscaled is None:
        # Try ncnn binary
        if _upscale_png_ncnn(input_png, output_png, scale=scale):
            return {
                "input": str(input_png), "output": str(output_png),
                "method": "ncnn-vulkan", "scale": scale,
                "input_res": f"{w0}×{h0}",
                "output_res": f"{w0*scale}×{h0*scale}",
            }
        # cv2 fallback
        upscaled = _upscale_png_cv2(img, scale)
        method = "cv2-lanczos4"

    h1, w1 = upscaled.shape[:2]
    out_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_png), out_bgr)

    return {
        "input": str(input_png), "output": str(output_png),
        "method": method, "scale": scale,
        "input_res": f"{w0}×{h0}", "output_res": f"{w1}×{h1}",
    }


# ---------------------------------------------------------------------------
# Video upscaler kept for backward compat (used by final QA, etc.)
# ---------------------------------------------------------------------------

def upscale_video_smart(input_path: Path, output_path: Path, *, scale: int = 2, use_realesrgan: bool = True) -> dict:
    """Upscale a video file using ffmpeg + LANCZOS (lightweight). For post-assembly pass."""
    ensure_dir(output_path.parent)
    h_factor, w_factor = scale, scale
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"scale=iw*{w_factor}:ih*{h_factor}:flags=lanczos",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-c:a", "copy", str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return {"method": "ffmpeg-lanczos", "scale": scale, "output": str(output_path)}


# ---------------------------------------------------------------------------
# CLI: upscale all scene PNGs from episode.scenes.json
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upscale scene PNGs with Real-ESRGAN 2×.")
    p.add_argument("--episode",  required=True, help="Path to episode.scenes.json")
    p.add_argument("--work-dir", required=True, help="Episode work directory")
    p.add_argument("--output",   required=True, help="Path to write episode.upscaled.json")
    p.add_argument("--scale",    type=int, default=2)
    p.add_argument("--no-realesrgan", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    progress_file = progress_path_from_env()
    episode_path  = Path(args.episode)
    work_dir      = ensure_dir(Path(args.work_dir))
    upscaled_dir  = ensure_dir(work_dir / "scenes_upscaled")
    output_path   = Path(args.output)

    write_stage_progress(progress_file, fraction=0.02, stage="upscale",
                         message="Starting Real-ESRGAN 2× on scene images…")

    episode = read_json(episode_path)
    shots = episode.get("shots", [])
    total = len(shots)

    manifest: list[dict] = []
    for i, shot in enumerate(shots):
        shot_id = shot.get("shot_id", f"s{i+1}")
        frac = 0.05 + 0.90 * (i / max(total, 1))
        write_stage_progress(progress_file, fraction=frac, stage="upscale",
                             message=f"Upscaling {shot_id} ({i+1}/{total})…")

        src = shot.get("scene_image", "")
        if not src or not Path(src).exists():
            print(f"[upscale] ⚠ {shot_id}: no scene image, skipping", flush=True)
            manifest.append({"shot_id": shot_id, "skipped": True})
            continue

        src_path = Path(src)
        out_path = upscaled_dir / f"{shot_id}_upscaled.png"

        try:
            report = upscale_png(src_path, out_path, scale=args.scale)
            shot["scene_image_upscaled"] = str(out_path)
            print(f"[upscale] ✓ {shot_id}: {report['input_res']} → {report['output_res']} ({report['method']})", flush=True)
            manifest.append({**report, "shot_id": shot_id})
        except Exception as exc:
            import warnings
            warnings.warn(f"[upscale] {shot_id} failed: {exc}")
            shot["scene_image_upscaled"] = src  # use original as fallback
            manifest.append({"shot_id": shot_id, "error": str(exc), "fallback": src})

    # Free memory
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    write_stage_progress(progress_file, fraction=0.98, stage="upscale",
                         message="Upscaling complete")

    episode["upscale_manifest"] = manifest
    write_json(output_path, episode)
    write_json(upscaled_dir / "upscale_manifest.json", manifest)
    print(f"[upscale] Done — {output_path}", flush=True)


if __name__ == "__main__":
    main()
