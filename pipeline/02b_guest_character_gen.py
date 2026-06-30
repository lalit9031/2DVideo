from __future__ import annotations

import argparse
from hashlib import sha1
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.bootstrap import bootstrap_guest_character
from pipeline.common import slugify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a guest character on demand.")
    parser.add_argument("--description", required=True, help="Short character description.")
    parser.add_argument("--source-episode", help="Episode that introduced the guest.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = slugify(args.description)
    suffix = sha1(args.description.encode("utf-8")).hexdigest()[:6]
    character_id = f"{base}_{suffix}"
    display_name = args.description.strip().title()
    bootstrap_guest_character(character_id, display_name, source_episode=args.source_episode)
    print(character_id)


if __name__ == "__main__":
    main()
