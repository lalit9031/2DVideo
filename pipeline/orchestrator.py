from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.bootstrap import bootstrap_demo_cast
from pipeline.common import OUTPUT_DIR, ROOT, ensure_dir, load_registry, today_iso, write_json
from pipeline.progress import progress_path_from_env, write_progress


@dataclass(frozen=True)
class Stage:
    name: str
    script: str


STAGES: tuple[Stage, ...] = (
    Stage("story", "pipeline/01_story_engine.py"),
    Stage("asset_check", "pipeline/02_asset_check.py"),
    Stage("tts", "pipeline/03_tts.py"),
    Stage("rig_render", "pipeline/04_rig_render.py"),
    Stage("broll", "pipeline/05_broll_gen.py"),
    Stage("assemble", "pipeline/07_assemble.py"),
    Stage("upscale", "pipeline/06_upscale.py"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the animation pipeline stages.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to generate.")
    parser.add_argument("--kind", choices=["poem", "story", "both"], default="poem")
    parser.add_argument("--bootstrap-demo", action="store_true", help="Create demo cast assets if missing.")
    parser.add_argument("--output-root", help="Override output root.")
    parser.add_argument("--template-poem", default="config/episode_templates/poem_template.json")
    parser.add_argument("--template-story", default="config/episode_templates/story_template.json")
    parser.add_argument("--progress-file", help="Path to write job progress JSON.")
    return parser


def _run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, script, *args]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Stage failed: {script}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def _template_for(kind: str, poem: str, story: str) -> str:
    return poem if kind == "poem" else story


def main() -> None:
    args = build_parser().parse_args()
    progress_file = Path(args.progress_file) if args.progress_file else progress_path_from_env()
    if args.bootstrap_demo or not load_registry():
        bootstrap_demo_cast()
    root = ensure_dir(Path(args.output_root) if args.output_root else OUTPUT_DIR / today_iso())
    kinds = ["poem", "story"] if args.kind == "both" else [args.kind]
    summaries = []
    total_steps = max(1, args.episodes * len(STAGES))
    completed_steps = 0
    write_progress(progress_file, percent=0.0, stage="bootstrap", message="Starting pipeline", status="running")
    for index in range(args.episodes):
        kind = kinds[index % len(kinds)]
        episode_id = f"{today_iso()}-{kind}-{index + 1:02d}"
        work_dir = ensure_dir(root / episode_id)
        episode_path = work_dir / "episode.json"
        validated_path = work_dir / "episode.validated.json"
        tts_path = work_dir / "episode.tts.json"
        rendered_path = work_dir / "episode.rendered.json"
        broll_path = work_dir / "episode.broll.json"
        assembled_path = work_dir / "assembled.mp4"
        final_path = work_dir / "final.mp4"
        template = _template_for(kind, args.template_poem, args.template_story)
        timings = []
        stage_specs = [
            ("story", [script_arg("--template", template), script_arg("--output", str(episode_path)), script_arg("--work-dir", str(work_dir)), script_arg("--episode-id", episode_id), script_arg("--kind", kind)]),
            ("asset_check", [script_arg("--episode", str(episode_path)), script_arg("--output", str(validated_path))]),
            ("tts", [script_arg("--episode", str(validated_path)), script_arg("--work-dir", str(work_dir / "tts")), script_arg("--output", str(tts_path))]),
            ("rig_render", [script_arg("--episode", str(tts_path)), script_arg("--work-dir", str(work_dir / "renders")), script_arg("--output", str(rendered_path))]),
            ("broll", [script_arg("--episode", str(rendered_path)), script_arg("--work-dir", str(work_dir / "broll")), script_arg("--output", str(broll_path))]),
            ("assemble", [script_arg("--episode", str(broll_path)), script_arg("--output", str(assembled_path))]),
            ("upscale", [script_arg("--input", str(assembled_path)), script_arg("--output", str(final_path))]),
        ]
        for stage_name, stage_args in stage_specs:
            write_progress(
                progress_file,
                percent=(completed_steps / total_steps) * 100.0,
                stage=stage_name,
                message=f"{episode_id}: starting {stage_name}",
                status="running",
            )
            started = perf_counter()
            _run(str(ROOT / stage_name_to_script(stage_name)), stage_args)
            timings.append({"stage": stage_name, "seconds": round(perf_counter() - started, 3)})
            completed_steps += 1
            write_progress(
                progress_file,
                percent=(completed_steps / total_steps) * 100.0,
                stage=stage_name,
                message=f"{episode_id}: finished {stage_name}",
                status="running",
            )
        summaries.append(
            {
                "episode_id": episode_id,
                "kind": kind,
                "work_dir": str(work_dir),
                "final_path": str(final_path),
                "timings": timings,
            }
        )
    write_json(root / "daily_summary.json", summaries)
    write_progress(progress_file, percent=100.0, stage="complete", message="Pipeline complete", status="done")
    print(root / "daily_summary.json")


def script_arg(flag: str, value: str) -> str:
    return f"{flag}={value}"

def stage_name_to_script(stage_name: str) -> str:
    return {
        "story": "pipeline/01_story_engine.py",
        "asset_check": "pipeline/02_asset_check.py",
        "tts": "pipeline/03_tts.py",
        "rig_render": "pipeline/04_rig_render.py",
        "broll": "pipeline/05_broll_gen.py",
        "assemble": "pipeline/07_assemble.py",
        "upscale": "pipeline/06_upscale.py",
    }[stage_name]


if __name__ == "__main__":
    main()
