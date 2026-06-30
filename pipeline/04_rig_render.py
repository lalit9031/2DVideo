from __future__ import annotations

import argparse

from pipeline.common import stage_not_implemented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render rigged puppet animation shots.")
    parser.add_argument("--episode", required=True, help="Path to episode JSON.")
    return parser


def main() -> None:
    build_parser().parse_args()
    stage_not_implemented("04_rig_render")


if __name__ == "__main__":
    main()

