from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ensure_dir, read_json, write_json
from pipeline.progress import progress_path_from_env, write_stage_progress

def render_broll(shot: dict, output_path: Path, progress_callback = None) -> bool:
    """
    Loop the shot's background image (upscaled or original) using ffmpeg to create a static video
    of the required duration.
    """
    import subprocess
    duration = float(shot.get("duration_sec", 4))
    img_path = shot.get("scene_image_upscaled") or shot.get("scene_image")
    ffmpeg_bin = "/home/lalit/Desktop/GPU optimization/Wan2GP/ffmpeg_bins/ffmpeg"
    if not img_path or not Path(img_path).exists():
        # Fallback: create a solid color screen if no image exists
        cmd = [
            ffmpeg_bin, "-y", "-threads", "4", "-f", "lavfi", "-i", "color=c=lightblue:s=1280x720:d=" + str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)
        ]
    else:
        # Loop the image
        cmd = [
            ffmpeg_bin, "-y", "-threads", "4", "-loop", "1", "-i", str(img_path),
            "-t", str(duration), "-r", "24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)
        ]
    
    if progress_callback:
        progress_callback(0.5)
        
    import os
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/home/lalit/Desktop/GPU optimization/Wan2GP/ffmpeg_bins:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        subprocess.run(cmd, check=True, capture_output=True, env=env)
        if progress_callback:
            progress_callback(1.0)
        return True
    except Exception as e:
        print(f"[broll] fallback failed: {e}", flush=True)
        return False



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate B-roll video for flagged shots.")
    parser.add_argument("--episode", required=True, help="Path to episode JSON.")
    parser.add_argument("--work-dir", help="Working directory for B-roll outputs.")
    parser.add_argument("--output", help="Optional output episode path.")
    parser.add_argument(
        "--ltx",
        action="store_true",
        help="Use LTX-Video (diffusers) for AI-generated B-roll. Falls back to procedural render if unavailable.",
    )
    parser.add_argument(
        "--ltx-steps",
        type=int,
        default=30,
        help="Number of LTX diffusion steps (default: 30; use 15 for faster/lower quality).",
    )
    parser.add_argument(
        "--ltx-guidance",
        type=float,
        default=3.5,
        help="LTX guidance scale (default: 3.5).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_path = Path(args.episode)
    episode = read_json(episode_path)
    work_dir = ensure_dir(Path(args.work_dir) if args.work_dir else episode_path.parent / "broll")
    broll_dir = ensure_dir(work_dir / "shots")
    manifests = []
    shots = [shot for shot in episode.get("shots", []) if shot.get("broll")]
    total_shots = max(1, len(shots))
    progress_file = progress_path_from_env()

    # Only import LTX helper when --ltx is requested (avoids slow diffusers import otherwise)
    ltx_generate = None
    if args.ltx:
        try:
            from pipeline.ltx_gen import generate_broll_ltx
            ltx_generate = generate_broll_ltx
            print("[broll] LTX-Video mode enabled.", flush=True)
        except ImportError as exc:
            print(f"[broll] WARNING: LTX import failed ({exc}). Using procedural fallback.", flush=True)

    for index, shot in enumerate(shots, start=1):
        shot_path = broll_dir / f"{shot['shot_id']}.mp4"
        duration = float(shot.get("duration_sec", 4))
        video_prompt = shot.get("video_prompt", "colourful abstract children's animation")
        used_ltx = False

        def report_shot_progress(fraction: float) -> None:
            write_stage_progress(
                progress_file,
                fraction=((index - 1) + fraction) / total_shots,
                stage="broll",
                message=f"B-roll {index}/{total_shots}: {shot['shot_id']}",
            )

        # ── Try LTX first ──────────────────────────────────────────────────
        if ltx_generate is not None and not shot_path.exists():
            print(f"[broll] LTX generating: {video_prompt[:80]}", flush=True)
            used_ltx = ltx_generate(
                prompt=video_prompt,
                output_path=shot_path,
                duration_sec=duration,
                num_inference_steps=args.ltx_steps,
                guidance_scale=args.ltx_guidance,
                progress_callback=report_shot_progress,
            )
            if used_ltx:
                print(f"[broll] LTX done → {shot_path.name}", flush=True)
            else:
                print("[broll] LTX returned False; falling back to procedural.", flush=True)

        # ── Procedural fallback ────────────────────────────────────────────
        if not used_ltx:
            result = render_broll(shot, shot_path, progress_callback=report_shot_progress)

        shot["rendered_video"] = str(shot_path)
        shot["broll_render_mode"] = "ltx" if used_ltx else "scenic_fallback"

        # Read actual frame count from produced file
        try:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(str(shot_path))
            fc = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            fp = cap.get(_cv2.CAP_PROP_FPS) or 24.0
            cap.release()
        except Exception:
            fc = int(round(duration * 24))
            fp = 24.0

        manifests.append(
            {
                "shot_id": shot["shot_id"],
                "video_path": str(shot_path),
                "frame_count": fc,
                "fps": fp,
                "render_mode": shot["broll_render_mode"],
            }
        )
        write_stage_progress(
            progress_file,
            fraction=index / total_shots,
            stage="broll",
            message=f"B-roll {index}/{total_shots}: {shot['shot_id']}",
        )

    episode["broll_manifest"] = manifests
    output_path = Path(args.output) if args.output else episode_path
    write_json(output_path, episode)
    write_json(work_dir / "broll_manifest.json", manifests)
    print(output_path)


if __name__ == "__main__":
    main()
