from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.assembly import assemble_episode_video
from pipeline.common import read_json, write_json
from pipeline.progress import progress_path_from_env, write_stage_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble final episode output.")
    parser.add_argument("--episode", required=True, help="Path to episode JSON.")
    parser.add_argument("--output", required=True, help="Path to final output video.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_path = Path(args.episode)
    episode = read_json(episode_path)
    progress_file = progress_path_from_env()
    write_stage_progress(progress_file, fraction=0.1, stage="assemble", message="Preparing audio mix")
    report = assemble_episode_video(episode, episode_path.parent, Path(args.output))
    audio_state = report.get("audio_mix_status", "unknown")
    write_stage_progress(progress_file, fraction=0.7, stage="assemble", message=f"Mixing audio ({audio_state})")
    write_json(Path(args.output).with_suffix(".manifest.json"), report)
    write_stage_progress(progress_file, fraction=1.0, stage="assemble", message="Assembly complete")
    print(args.output)


if __name__ == "__main__":
    main()
