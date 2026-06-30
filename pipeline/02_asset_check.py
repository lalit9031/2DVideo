from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.character_store import ensure_episode_characters
from pipeline.common import read_json, validate_json_schema, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate episode characters against registry.")
    parser.add_argument("--episode", required=True, help="Path to episode JSON.")
    parser.add_argument("--output", help="Optional output path for validated episode JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_path = Path(args.episode)
    episode = read_json(episode_path)
    validate_json_schema("episode", episode)
    episode = ensure_episode_characters(episode)
    validate_json_schema("episode", episode)
    output_path = Path(args.output) if args.output else episode_path
    write_json(output_path, episode)
    print(output_path)


if __name__ == "__main__":
    main()
