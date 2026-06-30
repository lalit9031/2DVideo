from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ensure_dir, read_json, write_json
from pipeline.story import build_episode, load_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate structured episode JSON.")
    parser.add_argument("--template", required=True, help="Path to episode template JSON.")
    parser.add_argument("--output", help="Path to write episode JSON.")
    parser.add_argument("--work-dir", help="Episode work directory. Defaults to output parent.")
    parser.add_argument("--episode-id", help="Override episode id.")
    parser.add_argument("--title", help="Override episode title.")
    parser.add_argument("--kind", choices=["poem", "story"], help="Override episode type.")
    parser.add_argument("--characters", nargs="*", help="Character ids to use.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    template = load_template(Path(args.template))
    episode = build_episode(
        template=template,
        episode_id=args.episode_id,
        title=args.title,
        kind=args.kind,
        characters=args.characters,
    )
    if args.work_dir:
        work_dir = ensure_dir(Path(args.work_dir))
        output_path = Path(args.output) if args.output else work_dir / "episode.json"
    else:
        output_path = Path(args.output) if args.output else Path(args.template).with_name("episode.json")
    write_json(output_path, episode)
    print(output_path)


if __name__ == "__main__":
    main()
