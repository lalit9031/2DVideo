"""
pipeline/orchestrator.py
~~~~~~~~~~~~~~~~~~~~~~~~
Main pipeline controller.

NEW 8-stage order:
  01 story        → episode.json  (Hindi script + SD prompts via Gemma 3 12B)
  02 asset_check  → episode.validated.json
  03 tts          → episode.tts.json
  04 scene_gen    → episode.scenes.json   (sd-server-vulkan, 20-25 steps)
  05 upscale      → episode.upscaled.json (Real-ESRGAN 2× PNGs)
  06 animatediff  → episode.animated.json (AnimateDiff-Lightning image→video)
  07 broll        → episode.broll.json    (LTX-2.3 for broll=true shots, optional)
  08 assemble     → assembled.mp4         (ffmpeg composite + audio + subs)
  qa              → final.qa_report.json

Format-aware:
  --format short  → 9:16 YouTube Shorts
  --format full   → 16:9 YouTube Full Length
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.bootstrap import bootstrap_demo_cast
from pipeline.common import (
    OUTPUT_DIR,
    ensure_dir,
    load_registry,
    today_iso,
    write_json,
)
from pipeline.progress import progress_path_from_env, read_progress, write_progress

ROOT = Path(__file__).resolve().parents[1]

STAGE_SCRIPTS = {
    "story":        "pipeline/01_story_engine.py",
    "asset_check":  "pipeline/02_asset_check.py",
    "tts":          "pipeline/03_tts.py",
    "scene_gen":    "pipeline/04_scene_gen.py",
    "upscale":      "pipeline/06_upscale.py",
    "animatediff":  "pipeline/06_animatediff.py",
    "broll":        "pipeline/05_broll_gen.py",
    "assemble":     "pipeline/07_assemble.py",
    "qa":           "pipeline/test_episode.py",
}

STAGES = [
    "story", "asset_check", "tts",
    "scene_gen", "upscale", "animatediff",
    "broll", "assemble", "qa"
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fruit Cartoon Pipeline Orchestrator")
    p.add_argument("--episodes",    type=int, default=1)
    p.add_argument("--kind",        choices=["poem", "story", "both"], default="poem")
    p.add_argument("--format",      choices=["short", "full"], default="full",
                   help="short=9:16 YouTube Shorts, full=16:9 YouTube Main")
    p.add_argument("--output-root", help="Override output root directory")
    p.add_argument("--template-poem",  default="config/episode_templates/poem_template.json")
    p.add_argument("--template-story", default="config/episode_templates/story_template.json")
    p.add_argument("--progress-file",  help="Path to write job progress JSON")
    p.add_argument("--use-ollama",     action="store_true",
                   help="Use Gemma 3 12B (Ollama) for prompts and story")
    p.add_argument("--no-broll",       action="store_true", help="Skip B-roll generation")
    p.add_argument("--no-realesrgan",  action="store_true", help="Skip Real-ESRGAN (use cv2)")
    p.add_argument("--ltx",            action="store_true", help="Use LTX-2.3 for B-roll")
    p.add_argument("--bootstrap-demo", action="store_true")
    return p


def _run(script: str, args: list[str], *, env: dict | None = None) -> None:
    cmd = [sys.executable, str(ROOT / script), *args]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Stage failed: {script}\nSTDOUT:\n{result.stdout[-600:]}\nSTDERR:\n{result.stderr[-600:]}"
        )


def _run_soft(script: str, args: list[str], *, env: dict | None = None) -> int:
    cmd = [sys.executable, str(ROOT / script), *args]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        print(
            f"[orchestrator] WARNING: {script} exited {result.returncode}\n"
            f"STDOUT: {result.stdout[:300]}\nSTDERR: {result.stderr[:300]}",
            flush=True,
        )
    return result.returncode


def _arg(flag: str, value: str) -> str:
    return f"{flag}={value}"


def main() -> None:
    args = build_parser().parse_args()
    progress_file = (
        Path(args.progress_file) if args.progress_file
        else (progress_path_from_env() or (OUTPUT_DIR / ".ui_progress.json"))
    )
    fmt = args.format
    current_stage = "bootstrap"

    try:
        if args.bootstrap_demo or not load_registry():
            bootstrap_demo_cast()

        root    = ensure_dir(Path(args.output_root) if args.output_root else OUTPUT_DIR / today_iso())
        kinds   = ["poem", "story"] if args.kind == "both" else [args.kind]
        summaries: list[dict] = []

        active_stages = [s for s in STAGES if not (s == "broll" and args.no_broll)]
        total_steps   = max(1, args.episodes * len(active_stages))
        completed     = 0

        write_progress(progress_file, percent=0.0, stage="bootstrap",
                       message="Starting pipeline", status="running")

        for index in range(args.episodes):
            kind       = kinds[index % len(kinds)]
            episode_id = f"{today_iso()}-{fmt}-{kind}-{index + 1:02d}"
            work_dir   = ensure_dir(root / episode_id)
            template   = args.template_poem if kind == "poem" else args.template_story

            # ── Intermediate file paths ──
            ep        = work_dir / "episode.json"
            validated = work_dir / "episode.validated.json"
            tts_json  = work_dir / "episode.tts.json"
            scenes_json   = work_dir / "episode.scenes.json"
            upscaled_json = work_dir / "episode.upscaled.json"
            animated_json = work_dir / "episode.animated.json"
            broll_json    = work_dir / "episode.broll.json"
            assembled_mp4 = work_dir / "assembled.mp4"
            final_mp4     = work_dir / "final.mp4"
            qa_json       = work_dir / "final.qa_report.json"

            # ── Stage argument lists ──
            story_args = [
                _arg("--template", template),
                _arg("--output",   str(ep)),
                _arg("--work-dir", str(work_dir)),
                _arg("--episode-id", episode_id),
                _arg("--kind", kind),
                _arg("--format", fmt),
            ]
            if args.use_ollama:
                story_args.append("--use-ollama")

            scene_args = [
                _arg("--episode",  str(tts_json)),
                _arg("--work-dir", str(work_dir / "scenes")),
                _arg("--output",   str(scenes_json)),
                _arg("--format",   fmt),
            ]
            if args.use_ollama:
                scene_args.append("--use-ollama")

            upscale_args = [
                _arg("--episode",  str(scenes_json)),
                _arg("--work-dir", str(work_dir / "scenes_upscaled")),
                _arg("--output",   str(upscaled_json)),
            ]
            if args.no_realesrgan:
                upscale_args.append("--no-realesrgan")

            animatediff_args = [
                _arg("--episode",  str(upscaled_json)),
                _arg("--work-dir", str(work_dir / "animated")),
                _arg("--output",   str(animated_json)),
            ]

            broll_args = [
                _arg("--episode",  str(animated_json)),
                _arg("--work-dir", str(work_dir / "broll")),
                _arg("--output",   str(broll_json)),
            ]
            if args.ltx:
                broll_args.append("--ltx")

            # After broll, the episode json includes broll enrichment
            assemble_input = broll_json if not args.no_broll else animated_json

            stage_specs = [
                ("story",       story_args),
                ("asset_check", [_arg("--episode", str(ep)), _arg("--output", str(validated))]),
                ("tts",         [_arg("--episode", str(validated)),
                                 _arg("--work-dir", str(work_dir / "tts")),
                                 _arg("--output",  str(tts_json))]),
                ("scene_gen",   scene_args),
                ("upscale",     upscale_args),
                ("animatediff", animatediff_args),
                ("broll",       broll_args),
                ("assemble",    [_arg("--episode", str(assemble_input)),
                                 _arg("--output",  str(assembled_mp4))]),
                ("qa",          [_arg("--episode", str(assemble_input)),
                                 _arg("--input",   str(assembled_mp4)),
                                 _arg("--output",  str(qa_json))]),
            ]

            timings: list[dict] = []
            for stage_name, stage_args in stage_specs:
                if stage_name not in active_stages:
                    continue
                current_stage = stage_name
                stage_start = (completed / total_steps) * 100.0
                stage_end   = ((completed + 1) / total_steps) * 100.0
                env = os.environ.copy()
                env["2DVIDEO_PROGRESS_FILE"]        = str(progress_file)
                env["2DVIDEO_STAGE_PROGRESS_START"] = str(stage_start)
                env["2DVIDEO_STAGE_PROGRESS_END"]   = str(stage_end)
                env["2DVIDEO_FORMAT"]               = fmt

                write_progress(progress_file, percent=stage_start, stage=stage_name,
                               message=f"{episode_id}: starting {stage_name}",
                               status="running")

                t0     = perf_counter()
                script = STAGE_SCRIPTS[stage_name]
                if stage_name == "qa":
                    rc = _run_soft(script, stage_args, env=env)
                    timings.append({"stage": stage_name, "seconds": round(perf_counter()-t0, 3), "exit_code": rc})
                else:
                    _run(script, stage_args, env=env)
                    timings.append({"stage": stage_name, "seconds": round(perf_counter()-t0, 3)})

                completed += 1
                write_progress(progress_file, percent=(completed/total_steps)*100.0,
                               stage=stage_name,
                               message=f"{episode_id}: finished {stage_name}",
                               status="running")
                print(f"[orchestrator] ✓ {stage_name} in {timings[-1]['seconds']:.1f}s", flush=True)

            # final.mp4 = assembled.mp4 (no separate upscale pass on video now)
            if assembled_mp4.exists() and not final_mp4.exists():
                assembled_mp4.rename(final_mp4)

            summaries.append({
                "episode_id": episode_id,
                "format": fmt,
                "kind": kind,
                "work_dir": str(work_dir),
                "final_path": str(final_mp4),
                "timings": timings,
            })

        write_json(root / "daily_summary.json", summaries)
        write_progress(progress_file, percent=100.0, stage="complete",
                       message="Pipeline complete ✓", status="done")
        print(root / "daily_summary.json")

    except Exception as exc:
        fp = read_progress(progress_file)
        write_progress(
            progress_file,
            percent=float(fp.get("percent", 0.0) or 0.0),
            stage=str(fp.get("stage", current_stage) or current_stage),
            message=str(exc),
            status="failed",
        )
        raise


if __name__ == "__main__":
    main()
