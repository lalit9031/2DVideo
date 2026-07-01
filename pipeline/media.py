from __future__ import annotations

import math
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from fractions import Fraction

import cv2
import numpy as np
import av

from pipeline.common import ensure_dir, slugify, write_json


SAMPLE_RATE = 22050


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_sec: float
    end_sec: float


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def has_pyav() -> bool:
    return av is not None


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
    try:
        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for frame in frames:
            vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return
    except Exception:
        pass

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


def extract_audio_to_wav(input_path: Path, output_path: Path) -> Path | None:
    try:
        container = av.open(str(input_path))
    except Exception:
        return None
    audio_stream = next((s for s in container.streams.audio), None)
    if audio_stream is None:
        container.close()
        return None
    ensure_dir(output_path.parent)
    samples: list[np.ndarray] = []
    sample_rate = audio_stream.rate or SAMPLE_RATE
    try:
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                pcm = frame.to_ndarray()
                if pcm.ndim > 1:
                    pcm = pcm[0]
                samples.append(np.asarray(pcm, dtype=np.int16))
    finally:
        container.close()
    merged = np.concatenate(samples) if samples else np.zeros(1, dtype=np.int16)
    with wave.open(str(output_path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(merged.tobytes())
    return output_path


def mux_video_audio(video_path: Path, audio_path: Path | None, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if audio_path is None or not audio_path.exists():
        frames, fps = read_video_frames(video_path)
        export_video(frames, output_path, fps)
        return
    video_container = av.open(str(video_path))
    video_stream = next((s for s in video_container.streams.video), None)
    if video_stream is None:
        video_container.close()
        raise RuntimeError(f"No video stream found in {video_path}")
    audio_container = av.open(str(audio_path))
    audio_stream = next((s for s in audio_container.streams.audio), None)
    if audio_stream is None:
        video_container.close()
        audio_container.close()
        raise RuntimeError(f"No audio stream found in {audio_path}")
    output = av.open(str(output_path), mode="w")
    out_video = output.add_stream("libx264", rate=Fraction(video_stream.average_rate or video_stream.base_rate or 24))
    out_video.width = video_stream.width
    out_video.height = video_stream.height
    out_video.pix_fmt = "yuv420p"
    out_video.options = {"crf": "18"}
    out_audio = output.add_stream("aac", rate=audio_stream.rate or SAMPLE_RATE)
    out_audio.layout = "mono"
    for packet in video_container.demux(video_stream):
        for frame in packet.decode():
            if frame.width != out_video.width or frame.height != out_video.height:
                frame = frame.reformat(width=out_video.width, height=out_video.height, format="rgb24")
            else:
                frame = frame.reformat(format="rgb24")
            for encoded in out_video.encode(frame):
                output.mux(encoded)
    for encoded in out_video.encode():
        output.mux(encoded)
    for packet in audio_container.demux(audio_stream):
        for frame in packet.decode():
            frame.sample_rate = audio_stream.rate or SAMPLE_RATE
            for encoded in out_audio.encode(frame):
                output.mux(encoded)
    for encoded in out_audio.encode():
        output.mux(encoded)
    output.close()
    video_container.close()
    audio_container.close()


def write_video_with_audio(frames: list[np.ndarray], audio_path: Path | None, output_path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("No frames supplied for video export.")
    ensure_dir(output_path.parent)
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    container = av.open(str(output_path), mode="w")
    vstream = container.add_stream("libx264", rate=fps)
    vstream.width = width
    vstream.height = height
    vstream.pix_fmt = "yuv420p"
    vstream.options = {"crf": "18"}
    for frame in frames:
        vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in vstream.encode(vf):
            container.mux(packet)
    for packet in vstream.encode():
        container.mux(packet)

    if audio_path is not None and audio_path.exists():
        with wave.open(str(audio_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            if channels != 1:
                raise ValueError("Only mono audio is supported in this pipeline.")
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        astream = container.add_stream("aac", rate=sample_rate)
        astream.layout = "mono"
        chunk = 1024
        for start in range(0, len(pcm), chunk):
            block = pcm[start:start + chunk]
            if block.size == 0:
                continue
            aframe = av.AudioFrame.from_ndarray(block.reshape(1, -1), format="s16", layout="mono")
            aframe.sample_rate = sample_rate
            for packet in astream.encode(aframe):
                container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)
    container.close()


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
