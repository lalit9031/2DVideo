from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Stage:
    name: str
    entrypoint: str


STAGES: tuple[Stage, ...] = (
    Stage("story", "pipeline/01_story_engine.py"),
    Stage("asset_check", "pipeline/02_asset_check.py"),
    Stage("tts", "pipeline/03_tts.py"),
    Stage("rig_render", "pipeline/04_rig_render.py"),
    Stage("upscale", "pipeline/06_upscale.py"),
    Stage("assemble", "pipeline/07_assemble.py"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the animation pipeline stages.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    return parser


def iter_stages() -> Iterable[Stage]:
    return STAGES


def main() -> None:
    build_parser().parse_args()
    raise NotImplementedError("orchestrator has not been implemented yet.")


if __name__ == "__main__":
    main()

