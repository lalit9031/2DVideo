from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ROOT, ensure_dir, load_voice_registry, read_json, write_json
from pipeline.media import SAMPLE_RATE, synthesize_voice_clip
from pipeline.progress import progress_path_from_env, write_stage_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize dialogue audio for an episode.")
    parser.add_argument("--episode", required=True, help="Path to episode JSON.")
    parser.add_argument("--work-dir", help="Working directory for TTS outputs.")
    parser.add_argument("--output", help="Optional output episode path.")
    return parser


def _resolve_reference_clip(voice_entry: dict) -> Path | None:
    path_str = voice_entry.get("reference_clip")
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        return path
    return None


def main() -> None:

    args = build_parser().parse_args()
    episode_path = Path(args.episode)
    episode = read_json(episode_path)
    voice_registry = load_voice_registry()
    work_dir = ensure_dir(Path(args.work_dir) if args.work_dir else episode_path.parent / "tts")
    audio_dir = ensure_dir(work_dir / "dialogue")
    manifest: list[dict] = []
    dialogue_items = [
        (shot_index, shot, line_index, line)
        for shot_index, shot in enumerate(episode.get("shots", []))
        for line_index, line in enumerate(shot.get("dialogue", []))
    ]
    total_items = max(1, len(dialogue_items))
    progress_file = progress_path_from_env()
    for item_index, (_, shot, idx, line) in enumerate(dialogue_items, start=1):
        voice_id = line["voice_id"]
        if voice_id not in voice_registry:
            raise KeyError(f"Missing voice registry entry for {voice_id}")
        voice_entry = voice_registry[voice_id]
        audio_path = audio_dir / f"{shot['shot_id']}_line_{idx + 1}.wav"
        ref_clip = _resolve_reference_clip(voice_entry)
        timings = synthesize_voice_clip(
            line["line"],
            audio_path,
            reference_clip_path=ref_clip,
            language=str(voice_entry.get("language", "en")),
            speed=float(voice_entry.get("speed", 1.0)),
        )
        line["audio_path"] = str(audio_path.relative_to(work_dir.parent if work_dir.parent != work_dir else work_dir))
        line["timings"] = [
            {"word": timing.word, "start_sec": timing.start_sec, "end_sec": timing.end_sec}
            for timing in timings
        ]
        manifest.append(
            {
                "shot_id": shot["shot_id"],
                "line_index": idx,
                "character": line["character"],
                "voice_id": voice_id,
                "audio_path": str(audio_path),
                "sample_rate": SAMPLE_RATE,
            }
        )
        write_stage_progress(
            progress_file,
            fraction=item_index / total_items,
            stage="tts",
            message=f"TTS {item_index}/{total_items}: {shot['shot_id']}",
        )
    episode["tts_manifest"] = manifest
    output_path = Path(args.output) if args.output else episode_path
    write_json(output_path, episode)
    write_json(work_dir / "tts_manifest.json", manifest)
    print(output_path)


if __name__ == "__main__":
    main()
