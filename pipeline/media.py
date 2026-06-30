from __future__ import annotations

import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from pipeline.common import ensure_dir, slugify, write_json


SAMPLE_RATE = 22050


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_sec: float
    end_sec: float


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def synthesize_voice_clip(
    text: str,
    output_path: Path,
    *,
    base_frequency: float,
    pitch_shift: float = 0.0,
    speed: float = 1.0,
) -> list[WordTiming]:
    words = [part for part in text.split() if part]
    if not words:
        words = ["..."]
    unit = max(0.18 / max(speed, 0.1), 0.06)
    timings: list[WordTiming] = []
    samples: list[np.ndarray] = []
    cursor = 0.0
    for index, word in enumerate(words):
        duration = max(unit + 0.01 * len(word), 0.09)
        duration /= max(speed, 0.1)
        start = cursor
        end = cursor + duration
        timings.append(WordTiming(word=word, start_sec=start, end_sec=end))
        cursor = end
        word_freq = base_frequency + (index % 5) * 35 + pitch_shift * 8.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        envelope = np.sin(np.linspace(0, math.pi, len(t))) ** 1.4
        tone = (
            0.28 * np.sin(2 * math.pi * word_freq * t)
            + 0.14 * np.sin(2 * math.pi * (word_freq * 1.5) * t)
        )
        pause = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)
        samples.append((tone * envelope).astype(np.float32))
        samples.append(pause)
    waveform = np.concatenate(samples) if samples else np.zeros(1, dtype=np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767).astype(np.int16)
    ensure_dir(output_path.parent)
    with wave.open(str(output_path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes(pcm.tobytes())
    return timings


def concatenate_wavs(wav_paths: Iterable[Path], output_path: Path) -> None:
    frames: list[np.ndarray] = []
    for path in wav_paths:
        with wave.open(str(path), "rb") as fh:
            pcm = fh.readframes(fh.getnframes())
            frames.append(np.frombuffer(pcm, dtype=np.int16))
    if frames:
        merged = np.concatenate(frames)
    else:
        merged = np.zeros(1, dtype=np.int16)
    ensure_dir(output_path.parent)
    with wave.open(str(output_path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes(merged.tobytes())


def export_video(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("No frames supplied for video export.")
    ensure_dir(output_path.parent)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {output_path}")
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def read_video_frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def upscale_frames(frames: list[np.ndarray], factor: float) -> list[np.ndarray]:
    if factor <= 1.0:
        return frames
    out: list[np.ndarray] = []
    for frame in frames:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)
        out.append(resized)
    return out


def write_manifest(path: Path, data: dict) -> None:
    write_json(path, data)


def safe_filename(value: str) -> str:
    return slugify(value).replace("-", "_")

