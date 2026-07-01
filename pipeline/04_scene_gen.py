"""
04_scene_gen.py — Per-shot AI Scene Image Generator
-----------------------------------------------------
Reads episode.tts.json (after TTS stage), calls sd-server-vulkan to
generate a scene PNG per shot (20-25 steps, DreamShaperXL_Lightning).
Writes scene PNGs to work_dir/scenes/ and produces episode.scenes.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ensure_dir, read_json, write_json
from pipeline.progress import progress_path_from_env, write_stage_progress
from pipeline.sd_client import start_server, stop_server, generate_image, dims_for_format
from pipeline.prompt_engine import enrich_episode_with_prompts

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STEPS = 22         # 20-25 range; 22 is the sweet spot for DreamShaperXL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate AI scene images per shot.")
    p.add_argument("--episode",   required=True, help="Path to episode.tts.json")
    p.add_argument("--work-dir",  required=True, help="Episode work directory")
    p.add_argument("--output",    required=True, help="Path to write episode.scenes.json")
    p.add_argument("--format",    choices=["short", "full"], default="full",
                   help="Video format: short (9:16) or full (16:9)")
    p.add_argument("--steps",     type=int, default=STEPS)
    p.add_argument("--use-ollama", action="store_true", help="Use Gemma 3 12B for prompts")
    p.add_argument("--ollama-model", default="gemma3:12b")
    p.add_argument("--ollama-url",   default="http://localhost:11434")
    return p


def main() -> None:
    args = build_parser().parse_args()
    progress_file = progress_path_from_env()
    episode_path  = Path(args.episode)
    work_dir      = ensure_dir(Path(args.work_dir))
    scenes_dir    = ensure_dir(work_dir / "scenes")
    output_path   = Path(args.output)
    fmt           = args.format

    write_stage_progress(progress_file, fraction=0.02, stage="scene_gen",
                         message="Loading episode data")

    episode = read_json(episode_path)
    fmt = fmt or episode.get("format", "full")

    # ── Step 1: Enrich episode with SD prompts (Gemma 3 12B or template) ──
    write_stage_progress(progress_file, fraction=0.05, stage="scene_gen",
                         message="Generating scene prompts via Gemma 3 12B…")
    episode = enrich_episode_with_prompts(
        episode,
        format=fmt,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )

    shots = episode.get("shots", [])
    total = len(shots)
    width, height = dims_for_format(fmt)

    # ── Step 2: Start sd-server-vulkan ──
    write_stage_progress(progress_file, fraction=0.08, stage="scene_gen",
                         message="Starting sd-server-vulkan (loading model)…")
    print("[scene_gen] Starting sd-server-vulkan…", flush=True)
    t_start = time.time()
    start_server()
    print(f"[scene_gen] Model loaded in {time.time()-t_start:.1f}s", flush=True)

    # ── Step 3: Generate scene image per shot ──
    scene_manifest: list[dict] = []
    for i, shot in enumerate(shots):
        shot_id = shot.get("shot_id", f"s{i+1}")
        frac = 0.10 + 0.80 * (i / max(total, 1))
        write_stage_progress(
            progress_file, fraction=frac, stage="scene_gen",
            message=f"Generating scene image {shot_id} ({i+1}/{total})…"
        )

        # Pure broll shots get a simple nature scene instead of a character scene
        if shot.get("broll") and not shot.get("characters"):
            prompt   = shot.get("video_prompt", "bright pastel children's garden, colorful flowers")
            prompt   += f", {args.format} format, Cocomelon kids cartoon style"
            negative = "ugly, blurry, dark, horror, realistic, photorealistic"
        else:
            prompt   = shot.get("image_prompt", "")
            negative = shot.get("negative_prompt", "")

        out_png = scenes_dir / f"{shot_id}.png"
        print(f"[scene_gen] Shot {shot_id}: {prompt[:70]}…", flush=True)
        t0 = time.time()

        try:
            generate_image(
                prompt=prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                steps=args.steps,
                cfg_scale=7.5,
                seed=-1,
                output_path=out_png,
                sampling_method="euler_a",
            )
            elapsed = time.time() - t0
            print(f"[scene_gen] ✓ {shot_id} done in {elapsed:.1f}s → {out_png.name}", flush=True)
            shot["scene_image"] = str(out_png)
            scene_manifest.append({
                "shot_id": shot_id,
                "scene_image": str(out_png),
                "prompt": prompt[:120],
                "elapsed_sec": round(elapsed, 2),
                "width": width,
                "height": height,
            })
        except Exception as exc:
            log.error("[scene_gen] Failed shot %s: %s", shot_id, exc)
            shot["scene_image"] = ""
            scene_manifest.append({
                "shot_id": shot_id, "scene_image": "", "error": str(exc)
            })

    # ── Step 4: Stop server ──
    stop_server()
    write_stage_progress(progress_file, fraction=0.98, stage="scene_gen",
                         message="Scene generation complete")

    episode["scene_manifest"] = scene_manifest
    write_json(output_path, episode)
    write_json(scenes_dir / "scene_manifest.json", scene_manifest)
    print(f"[scene_gen] Done — {len(scene_manifest)} scenes → {output_path}", flush=True)


if __name__ == "__main__":
    main()
