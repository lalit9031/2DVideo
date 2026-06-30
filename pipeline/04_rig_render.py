from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ensure_dir, read_json, write_json
from pipeline.rendering import FPS, render_shot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render rigged puppet animation shots.")
    parser.add_argument("--episode", required=True, help="Path to episode JSON.")
    parser.add_argument("--work-dir", help="Working directory for shot renders.")
    parser.add_argument("--output", help="Optional output episode path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_path = Path(args.episode)
    episode = read_json(episode_path)
    work_dir = ensure_dir(Path(args.work_dir) if args.work_dir else episode_path.parent / "renders")
    shots_dir = ensure_dir(work_dir / "shots")
    manifests = []
    for shot in episode.get("shots", []):
        if shot.get("broll"):
            continue
        shot_path = shots_dir / f"{shot['shot_id']}.mp4"
        result = render_shot(episode, shot, shot_path, fps=FPS)
        shot["rendered_video"] = str(shot_path)
        manifests.append(
            {
                "shot_id": result.shot_id,
                "video_path": str(result.video_path),
                "frame_count": result.frame_count,
                "fps": result.fps,
            }
        )
    episode["render_manifest"] = manifests
    output_path = Path(args.output) if args.output else episode_path
    write_json(output_path, episode)
    write_json(work_dir / "render_manifest.json", manifests)
    print(output_path)


if __name__ == "__main__":
    main()
