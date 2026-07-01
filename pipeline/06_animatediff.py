"""
06_animatediff.py — AnimateDiff-Lightning Image-to-Video per Shot
------------------------------------------------------------------
Takes each upscaled scene PNG and animates it using AnimateDiff-Lightning
with the same prompt + negative_prompt, producing a 2-second MP4 per shot.

Requirements:
    pip install diffusers transformers accelerate safetensors imageio
    pip install imageio[ffmpeg]

AnimateDiff-Lightning:
  - Model: ByteDance/AnimateDiff-Lightning
  - Motion module: animatediff_lightning_4step_diffusers.safetensors
  - Base SD1.5 checkpoint: epiCRealism or DreamShaper (1.5 variant)
  - Steps: 4 (distilled Lightning steps — very fast)
  - Output: 16 frames at 8 fps = 2 seconds
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ensure_dir, read_json, write_json
from pipeline.progress import progress_path_from_env, write_stage_progress

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Fix ROCm memory fragmentation
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

FRAMES = 16         # 16 frames @ 8fps = 2 seconds per shot
FPS    = 8
STEPS  = 4          # AnimateDiff-Lightning distilled steps

# AnimateDiff-Lightning model IDs
MOTION_ADAPTER_ID = "ByteDance/AnimateDiff-Lightning"
MOTION_CKPT       = "animatediff_lightning_4step_diffusers.safetensors"
# SD1.5 base for anime/cartoon look — best with AnimateDiff-Lightning
BASE_MODEL_ID     = "Lykon/dreamshaper-8"


def _load_pipeline(device: str):
    """Load AnimateDiff-Lightning pipeline (cached after first load)."""
    import torch
    from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    dtype = torch.float16 if device != "cpu" else torch.float32

    log.info("[animatediff] Loading base motion adapter config (v1-5)…")
    config = MotionAdapter.load_config("guoyww/animatediff-motion-adapter-v1-5")
    
    # Update config to match AnimateDiff-Lightning layout
    if isinstance(config, tuple):
        config_dict = dict(config[0])
    else:
        config_dict = dict(config)
    config_dict["motion_max_seq_length"] = 32
    config_dict["use_motion_mid_block"] = True

    adapter = MotionAdapter.from_config(config_dict).to(device, dtype)

    log.info("[animatediff] Downloading AnimateDiff-Lightning weights…")
    ckpt_path = hf_hub_download(
        repo_id="ByteDance/AnimateDiff-Lightning",
        filename="animatediff_lightning_4step_diffusers.safetensors"
    )

    log.info("[animatediff] Loading Lightning weights state_dict…")
    adapter.load_state_dict(load_file(ckpt_path, device=device))

    log.info("[animatediff] Loading base pipeline %s…", BASE_MODEL_ID)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        BASE_MODEL_ID,
        motion_adapter=adapter,
        torch_dtype=dtype,
    )
    from diffusers import DDIMScheduler
    # DDIMScheduler avoids the Euler ROCm indexing bug entirely
    pipe.scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        clip_sample=False,
        steps_offset=1
    )

    pipe.to(device)
    pipe.enable_vae_slicing()
    if device != "cpu":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
    return pipe


def _animate_shot(
    pipe,
    *,
    image_path: Path,
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    frames: int = FRAMES,
    fps: int = FPS,
    steps: int = STEPS,
    device: str,
) -> bool:
    """Animate a single shot PNG → MP4. Returns True on success."""
    import torch
    from PIL import Image
    from diffusers.utils import export_to_video

    if not image_path.exists():
        log.warning("[animatediff] scene image missing: %s", image_path)
        return False

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    dtype = torch.float16 if device != "cpu" else torch.float32

    log.info("[animatediff] Animating %s (%dx%d) → %s", image_path.name, w, h, output_path.name)
    t0 = time.time()

    # SD1.5 AnimateDiff generation size (must be small to fit in VRAM, e.g. 512x288 or 288x512)
    # Check aspect ratio
    if w >= h:
        gen_w, gen_h = 512, 288
    else:
        gen_w, gen_h = 288, 512

    # Resize source image to generation dimensions
    resized_img = img.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
    video_frames_input = [resized_img] * frames

    with torch.inference_mode():
        result = pipe(
            video=video_frames_input,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=1.0,         # Lightning needs no CFG (guidance_scale=1.0)
            num_inference_steps=steps,
            strength=0.75,
            generator=torch.Generator(device=device).manual_seed(42),
        )

    video_frames = result.frames[0]   # list of PIL Images
    
    # Scale frames back to the original upscaled resolution (e.g. 2K)
    log.info("[animatediff] Resizing frames from %dx%d to %dx%d...", gen_w, gen_h, w, h)
    resized_frames = [frame.resize((w, h), Image.Resampling.LANCZOS) for frame in video_frames]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(resized_frames, str(output_path), fps=fps)

    elapsed = time.time() - t0
    log.info("[animatediff] ✓ %s done in %.1fs", output_path.name, elapsed)
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Animate scene PNGs with AnimateDiff-Lightning.")
    p.add_argument("--episode",  required=True, help="Path to episode.upscaled.json")
    p.add_argument("--work-dir", required=True, help="Episode work directory")
    p.add_argument("--output",   required=True, help="Path to write episode.animated.json")
    p.add_argument("--frames",   type=int, default=FRAMES)
    p.add_argument("--fps",      type=int, default=FPS)
    p.add_argument("--steps",    type=int, default=STEPS)
    p.add_argument("--cpu",      action="store_true", help="Force CPU mode")
    return p


def main() -> None:
    args = build_parser().parse_args()
    progress_file = progress_path_from_env()
    episode_path  = Path(args.episode)
    work_dir      = ensure_dir(Path(args.work_dir))
    clips_dir     = ensure_dir(work_dir / "animated")
    output_path   = Path(args.output)

    write_stage_progress(progress_file, fraction=0.02, stage="animatediff",
                         message="Loading AnimateDiff-Lightning model…")

    episode = read_json(episode_path)
    shots = episode.get("shots", [])
    total = len(shots)

    # Device
    try:
        import torch
        device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"

    print(f"[animatediff] Device: {device}, shots: {total}", flush=True)

    # Load model once
    pipe = _load_pipeline(device)
    print("[animatediff] Pipeline loaded.", flush=True)

    manifest: list[dict] = []
    for i, shot in enumerate(shots):
        shot_id = shot.get("shot_id", f"s{i+1}")
        frac = 0.10 + 0.85 * (i / max(total, 1))
        write_stage_progress(
            progress_file, fraction=frac, stage="animatediff",
            message=f"Animating shot {shot_id} ({i+1}/{total})…"
        )

        # Use upscaled image if available, else original scene image
        img_path = Path(shot.get("scene_image_upscaled") or shot.get("scene_image", ""))
        prompt   = shot.get("image_prompt") or shot.get("video_prompt", "colorful garden scene")
        negative = shot.get("negative_prompt", "ugly, blurry, dark, horror")
        out_mp4  = clips_dir / f"{shot_id}.mp4"

        t0 = time.time()
        ok = _animate_shot(
            pipe,
            image_path=img_path,
            prompt=prompt,
            negative_prompt=negative,
            output_path=out_mp4,
            frames=args.frames,
            fps=args.fps,
            steps=args.steps,
            device=device,
        )
        elapsed = time.time() - t0

        if ok:
            shot["animated_clip"] = str(out_mp4)
            manifest.append({
                "shot_id": shot_id,
                "animated_clip": str(out_mp4),
                "frames": args.frames,
                "fps": args.fps,
                "elapsed_sec": round(elapsed, 2),
            })
            print(f"[animatediff] ✓ {shot_id} → {out_mp4.name} ({elapsed:.1f}s)", flush=True)
        else:
            # Fallback: create a static video from the image using ffmpeg
            if img_path.exists():
                import subprocess
                duration = shot.get("duration_sec", 3)
                cmd = [
                    "ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
                    "-t", str(duration), "-r", str(args.fps),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    shot["animated_clip"] = str(out_mp4)
                    manifest.append({"shot_id": shot_id, "animated_clip": str(out_mp4), "method": "ffmpeg-static"})
                    print(f"[animatediff] ⚠ {shot_id} used static fallback", flush=True)
                except Exception as e:
                    log.error("[animatediff] Fallback failed for %s: %s", shot_id, e)
                    shot["animated_clip"] = ""
                    manifest.append({"shot_id": shot_id, "animated_clip": "", "error": str(e)})
            else:
                shot["animated_clip"] = ""
                manifest.append({"shot_id": shot_id, "animated_clip": "", "error": "no image"})

    # Free GPU memory
    del pipe
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    write_stage_progress(progress_file, fraction=0.98, stage="animatediff",
                         message="AnimateDiff complete")

    episode["animated_manifest"] = manifest
    write_json(output_path, episode)
    write_json(clips_dir / "animated_manifest.json", manifest)
    print(f"[animatediff] Done — {len(manifest)} clips → {output_path}", flush=True)


if __name__ == "__main__":
    main()
