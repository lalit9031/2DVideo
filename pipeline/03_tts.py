from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ensure_dir, load_voice_registry, read_json, write_json
from pipeline.media import SAMPLE_RATE, synthesize_voice_clip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize dialogue audio for an episode.")
    parser.add_argument("--episode", required=True, help="Path to episode JSON.")
    parser.add_argument("--work-dir", help="Working directory for TTS outputs.")
    parser.add_argument("--output", help="Optional output episode path.")
    return parser


def _voice_frequency(voice_id: str) -> float:
    seed = sum(ord(ch) for ch in voice_id)
    return 150.0 + (seed % 120)


def main() -> None:
    args = build_parser().parse_args()
    episode_path = Path(args.episode)
    episode = read_json(episode_path)
    voice_registry = load_voice_registry()
    work_dir = ensure_dir(Path(args.work_dir) if args.work_dir else episode_path.parent / "tts")
    audio_dir = ensure_dir(work_dir / "dialogue")
    manifest: list[dict] = []
    for shot in episode.get("shots", []):
        for idx, line in enumerate(shot.get("dialogue", [])):
            voice_id = line["voice_id"]
            if voice_id not in voice_registry:
                raise KeyError(f"Missing voice registry entry for {voice_id}")
            voice_entry = voice_registry[voice_id]
            audio_path = audio_dir / f"{shot['shot_id']}_line_{idx + 1}.wav"
            timings = synthesize_voice_clip(
                line["line"],
                audio_path,
                base_frequency=_voice_frequency(voice_id),
                pitch_shift=float(voice_entry.get("pitch_shift", 0)),
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
    episode["tts_manifest"] = manifest
    output_path = Path(args.output) if args.output else episode_path
    write_json(output_path, episode)
    write_json(work_dir / "tts_manifest.json", manifest)
    print(output_path)


if __name__ == "__main__":
    main()
