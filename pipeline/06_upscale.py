from __future__ import annotations

import argparse

from pipeline.common import stage_not_implemented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upscale rendered output.")
    parser.add_argument("--input", required=True, help="Path to source video.")
    parser.add_argument("--output", required=True, help="Path to upscaled video.")
    return parser


def main() -> None:
    build_parser().parse_args()
    stage_not_implemented("06_upscale")


if __name__ == "__main__":
    main()

