from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from pipeline.common import ensure_dir, read_json, write_json
from pipeline.media import SAMPLE_RATE, concatenate_wavs, export_video, read_video_frames, upscale_frames
import wave


def _collect_audio_paths(episode: dict, episode_root: Path) -> list[Path]:
    paths: list[Path] = []
    for shot in episode.get("shots", []):
        for line in shot.get("dialogue", []):
            audio_path = line.get("audio_path")
            if audio_path:
                path = Path(audio_path)
                if not path.is_absolute():
                    path = episode_root / audio_path
                paths.append(path)
    return paths


def assemble_episode_video(
    episode: dict,
    episode_root: Path,
    output_path: Path,
    *,
    upscaled: bool = False,
) -> dict:
    shot_paths: list[Path] = []
    for shot in episode.get("shots", []):
        rendered = shot.get("rendered_video")
        if rendered:
            path = Path(rendered)
            if not path.is_absolute():
                path = episode_root / rendered
            shot_paths.append(path)
    frames: list[np.ndarray] = []
    fps = None
    for path in shot_paths:
        shot_frames, shot_fps = read_video_frames(path)
        if fps is None:
            fps = shot_fps
        frames.extend(shot_frames)
    if not frames:
        raise ValueError("No rendered frames were found for assembly.")
    ensure_dir(output_path.parent)
    export_video(frames, output_path, fps or 12.0)
    audio_paths = _collect_audio_paths(episode, episode_root)
    final_audio = output_path.with_suffix(".wav")
    if audio_paths:
        concatenate_wavs(audio_paths, final_audio)
    else:
        ensure_dir(final_audio.parent)
        with wave.open(str(final_audio), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(SAMPLE_RATE)
            fh.writeframes(np.zeros(SAMPLE_RATE // 2, dtype=np.int16).tobytes())
    return {
        "final_video": str(output_path),
        "final_audio": str(final_audio),
        "shot_count": len(shot_paths),
        "frame_count": len(frames),
        "fps": fps or 12.0,
        "upscaled": upscaled,
    }


def upscale_video_file(input_path: Path, output_path: Path, factor: float = 2.0) -> dict:
    frames, fps = read_video_frames(input_path)
    upscaled = upscale_frames(frames, factor)
    export_video(upscaled, output_path, fps)
    return {"input": str(input_path), "output": str(output_path), "factor": factor, "fps": fps}
