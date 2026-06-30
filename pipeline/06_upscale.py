from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.assembly import upscale_video_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upscale rendered output.")
    parser.add_argument("--input", required=True, help="Path to source video.")
    parser.add_argument("--output", required=True, help="Path to upscaled video.")
    parser.add_argument("--factor", type=float, default=2.0, help="Upscale factor.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = upscale_video_file(Path(args.input), Path(args.output), factor=args.factor)
    print(report["output"])


if __name__ == "__main__":
    main()
