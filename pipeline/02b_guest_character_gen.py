from __future__ import annotations

import argparse

from pipeline.common import stage_not_implemented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a guest character on demand.")
    parser.add_argument("--description", required=True, help="Short character description.")
    return parser


def main() -> None:
    build_parser().parse_args()
    stage_not_implemented("02b_guest_character_gen")


if __name__ == "__main__":
    main()

