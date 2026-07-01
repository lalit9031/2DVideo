from __future__ import annotations

from pathlib import Path

import numpy as np
import wave

from pipeline.common import ensure_dir, write_json
from pipeline.media import SAMPLE_RATE, export_video, extract_audio_to_wav, read_video_frames, upscale_frames, write_video_with_audio


def _collect_audio_items(episode: dict, episode_root: Path) -> list[dict]:
    items: list[dict] = []
    for shot in episode.get("shots", []):
        for line_index, line in enumerate(shot.get("dialogue", []), start=1):
            audio_path = line.get("audio_path")
            resolved: Path | None = None
            exists = False
            if audio_path:
                path = Path(audio_path)
                if not path.is_absolute():
                    path = episode_root / audio_path
                resolved = path
                exists = path.exists()
            items.append(
                {
                    "shot_id": shot.get("shot_id", ""),
                    "line_index": line_index,
                    "character": line.get("character", ""),
                    "audio_path": audio_path or "",
                    "resolved_audio_path": str(resolved) if resolved else "",
                    "exists": exists,
                }
            )
    return items


def _read_wav_samples(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        sample_rate = fh.getframerate()
        channels = fh.getnchannels()
        if channels != 1:
            raise ValueError("Only mono audio is supported in this pipeline.")
        pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
    return pcm, sample_rate


def _write_wav_samples(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    ensure_dir(path.parent)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def _build_episode_audio(episode: dict, episode_root: Path, target_duration_sec: float, output_path: Path) -> Path:
    shot_audio_parts: list[np.ndarray] = []
    sample_rate = SAMPLE_RATE
    for shot in episode.get("shots", []):
        shot_audio_paths: list[Path] = []
        for line in shot.get("dialogue", []):
            audio_path = line.get("audio_path")
            if not audio_path:
                continue
            path = Path(audio_path)
            if not path.is_absolute():
                path = episode_root / audio_path
            shot_audio_paths.append(path)
        shot_samples: list[np.ndarray] = []
        for path in shot_audio_paths:
            samples, sample_rate = _read_wav_samples(path)
            if samples.size:
                shot_samples.append(samples)
        shot_target = max(1, int(round(float(shot.get("duration_sec", 0)) * sample_rate)))
        if shot_samples:
            base = np.concatenate(shot_samples)
            if base.size >= shot_target:
                extended = base[:shot_target]
            else:
                repeats = max(1, int(np.ceil(shot_target / max(base.size, 1))))
                extended = np.tile(base, repeats)[:shot_target]
        else:
            t = np.linspace(0.0, float(shot.get("duration_sec", 0)), shot_target, endpoint=False, dtype=np.float32)
            seed = sum(ord(ch) for ch in shot.get("shot_id", "shot"))
            phase = (seed % 360) * np.pi / 180.0
            ambient = 0.03 * np.sin(2 * np.pi * 110.0 * t + phase) + 0.02 * np.sin(2 * np.pi * 220.0 * t * 0.5 + phase / 2.0)
            ambient += 0.01 * np.sin(2 * np.pi * 37.0 * t)
            extended = np.clip(ambient * 32767.0, -32768, 32767).astype(np.int16)
        shot_audio_parts.append(np.asarray(extended, dtype=np.int16))
    if shot_audio_parts:
        combined = np.concatenate(shot_audio_parts)
    else:
        combined = np.zeros(max(1, int(round(target_duration_sec * sample_rate))), dtype=np.int16)
    target_samples = max(1, int(round(target_duration_sec * sample_rate)))
    if combined.size < target_samples:
        combined = np.tile(combined if combined.size else np.zeros(1, dtype=np.int16), int(np.ceil(target_samples / max(combined.size, 1))))[:target_samples]
    else:
        combined = combined[:target_samples]
    _write_wav_samples(output_path, combined, sample_rate)
    return output_path


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
    final_audio = output_path.with_suffix(".wav")
    audio_items = _collect_audio_items(episode, episode_root)
    missing_audio = [item for item in audio_items if item["audio_path"] and not item["exists"]]
    _build_episode_audio(episode, episode_root, float(episode.get("target_duration_sec", len(frames) / (fps or 24.0))), final_audio)
    write_video_with_audio(frames, final_audio, output_path, fps or 24.0)
    return {
        "final_video": str(output_path),
        "final_audio": str(final_audio),
        "shot_count": len(shot_paths),
        "frame_count": len(frames),
        "fps": fps or 24.0,
        "upscaled": upscaled,
        "audio_items": audio_items,
        "audio_missing_count": len(missing_audio),
        "audio_mix_status": "missing" if not audio_items else ("partial" if missing_audio else "ready"),
    }


def upscale_video_file(input_path: Path, output_path: Path, factor: float = 2.0) -> dict:
    frames, fps = read_video_frames(input_path)
    upscaled = upscale_frames(frames, factor)
    temp_audio = output_path.with_suffix(".wav")
    extract_audio_to_wav(input_path, temp_audio)
    write_video_with_audio(upscaled, temp_audio if temp_audio.exists() else None, output_path, fps or 24.0)
    try:
        if temp_audio.exists():
            temp_audio.unlink()
    except Exception:
        pass
    return {"input": str(input_path), "output": str(output_path), "factor": factor, "fps": fps or 24.0}
