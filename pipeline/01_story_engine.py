from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ensure_dir, read_json, write_json
from pipeline.story import build_episode, build_episode_with_ollama, load_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate structured episode JSON.")
    parser.add_argument("--template", required=True, help="Path to episode template JSON.")
    parser.add_argument("--output", help="Path to write episode JSON.")
    parser.add_argument("--work-dir", help="Episode work directory. Defaults to output parent.")
    parser.add_argument("--episode-id", help="Override episode id.")
    parser.add_argument("--title", help="Override episode title.")
    parser.add_argument("--kind", choices=["poem", "story"], help="Override episode type.")
    parser.add_argument("--format", choices=["short", "full"], default="full",
                        help="Video format: short=9:16 YouTube Shorts, full=16:9 YouTube Main")
    parser.add_argument("--characters", nargs="*", help="Character ids to use.")
    # LLM options
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama LLM to generate richer dialogue.")
    parser.add_argument("--ollama-model", default="gemma3:12b", help="Ollama model name (default: gemma3:12b).")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    template = load_template(Path(args.template))
    builder = build_episode_with_ollama if args.use_ollama else build_episode
    episode_kwargs: dict = dict(
        template=template,
        episode_id=args.episode_id,
        title=args.title,
        kind=args.kind,
        characters=args.characters,
    )
    if args.use_ollama:
        episode_kwargs["model"] = args.ollama_model
        episode_kwargs["ollama_url"] = args.ollama_url
    episode = builder(**episode_kwargs)
    # Inject format into episode JSON for downstream stages
    episode["format"] = getattr(args, "format", "full")
    if args.work_dir:
        work_dir = ensure_dir(Path(args.work_dir))
        output_path = Path(args.output) if args.output else work_dir / "episode.json"
    else:
        output_path = Path(args.output) if args.output else Path(args.template).with_name("episode.json")
    write_json(output_path, episode)
    print(output_path)


if __name__ == "__main__":
    main()
