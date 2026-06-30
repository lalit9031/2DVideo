from __future__ import annotations

import argparse

from pipeline.common import stage_not_implemented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate structured episode JSON.")
    parser.add_argument("--template", required=True, help="Path to episode template JSON.")
    parser.add_argument("--output", required=True, help="Path to write episode JSON.")
    return parser


def main() -> None:
    build_parser().parse_args()
    stage_not_implemented("01_story_engine")


if __name__ == "__main__":
    main()

