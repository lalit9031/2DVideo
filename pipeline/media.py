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


# ---------------------------------------------------------------------------
# XTTS-v2 integration helpers
# ---------------------------------------------------------------------------

_TTS_MODEL: object | None = None


def _load_tts_model() -> object | None:
    global _TTS_MODEL
    if _TTS_MODEL is not None:
        return _TTS_MODEL
    # Auto-accept the CPML non-commercial license for unattended operation.
    import os
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    try:
        from TTS.api import TTS

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _TTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        return _TTS_MODEL
    except Exception as exc:
        import warnings
        warnings.warn(f"XTTS-v2 model could not be loaded: {exc}. Will use fallback sine-wave TTS.")
        return None


def _chunk_text(text: str, max_chars: int = 250) -> list[str]:
    """Split text into chunks of at most max_chars, preferring sentence boundaries."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sent) <= max_chars:
                current = sent
            else:
                words = sent.split()
                current = ""
                for w in words:
                    if len(current) + len(w) + 1 > max_chars:
                        if current:
                            chunks.append(current)
                        current = w
                    else:
                        current = (current + " " + w).strip()
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


def _synthesize_with_xtts(
    text: str,
    output_path: Path,
    reference_clip_path: Path | str,
    language: str = "en",
    speed: float = 1.0,
) -> None:
    """Synthesize speech using XTTS-v2, chunking long text."""
    model = _load_tts_model()
    if model is None:
        _synthesize_fallback(text, output_path, speed)
        return

    import tempfile

    chunks = _chunk_text(text)
    audio_chunks: list[bytes] = []

    for chunk in chunks:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # XTTS-v2 does not accept a speed parameter; ignore speed for XTTS.
            model.tts_to_file(
                text=chunk,
                speaker_wav=str(reference_clip_path),
                language=language,
                file_path=tmp_path,
            )
            with wave.open(tmp_path, "rb") as fh:
                audio_chunks.append(fh.readframes(fh.getnframes()))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if not audio_chunks:
        _synthesize_fallback(text, output_path, speed)
        return

    # Concatenate all chunk frames, resampling to the project's standard SAMPLE_RATE.
    raw = b"".join(audio_chunks)
    pcm = np.frombuffer(raw, dtype=np.int16)

    # XTTS-v2 outputs at 24000 Hz; resample to the project-wide SAMPLE_RATE (22050).
    target_samples = int(round(len(pcm) * SAMPLE_RATE / 24000))
    if target_samples != len(pcm):
        x = np.linspace(0, 1, max(1, len(pcm)))
        x_new = np.linspace(0, 1, max(1, target_samples))
        pcm = np.interp(x_new, x, pcm.astype(np.float32)).astype(np.int16)

    ensure_dir(output_path.parent)
    with wave.open(str(output_path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes(pcm.tobytes())


def _synthesize_fallback(text: str, output_path: Path, speed: float = 1.0) -> None:
    """Fallback sine-wave TTS when XTTS is unavailable."""
    _synthesize_fallback_with_seed(text, output_path, speed, voice_seed=None)


def _synthesize_fallback_with_seed(
    text: str,
    output_path: Path,
    speed: float = 1.0,
    voice_seed: int | None = None,
) -> None:
    """Fallback sine-wave TTS with a seedable tone profile."""
    words = [part for part in text.split() if part]
    if not words:
        words = ["..."]
    unit = max(0.18 / max(speed, 0.1), 0.06)
    samples: list[np.ndarray] = []
    seed = abs(int(voice_seed or 0))
    base_freq = 168.0 + (seed % 11) * 17.0
    harmonic_ratio = 1.35 + (seed % 7) * 0.05
    breath_gap = 0.012 + (seed % 5) * 0.003
    for index, word in enumerate(words):
        duration = max(unit + 0.01 * len(word), 0.09)
        duration /= max(speed, 0.1)
        freq = base_freq + (index % 5) * (24 + (seed % 3) * 3)
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        envelope = np.sin(np.linspace(0, math.pi, len(t))) ** 1.4
        tone = (
            0.28 * np.sin(2 * math.pi * freq * t)
            + 0.14 * np.sin(2 * math.pi * (freq * harmonic_ratio) * t)
        )
        pause = np.zeros(int(SAMPLE_RATE * breath_gap), dtype=np.float32)
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


def _wav_duration_sec(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as fh:
            frames = fh.getnframes()
            rate = fh.getframerate()
            return frames / max(rate, 1)
    except Exception:
        return 0.0


def synthesize_voice_clip(
    text: str,
    output_path: Path,
    *,
    reference_clip_path: Path | str | None = None,
    language: str = "en",
    speed: float = 1.0,
    voice_seed: int | None = None,
) -> list[WordTiming]:
    words = [part for part in text.split() if part]
    if not words:
        words = ["..."]

    if reference_clip_path is not None and Path(str(reference_clip_path)).exists():
        _synthesize_with_xtts(text, output_path, reference_clip_path, language, speed)
    else:
        _synthesize_fallback_with_seed(text, output_path, speed, voice_seed=voice_seed)

    total_duration = _wav_duration_sec(output_path)
    if total_duration <= 0:
        total_duration = 0.5 * len(words)

    char_total = sum(len(w) for w in words)
    timings: list[WordTiming] = []
    cursor = 0.0
    for word in words:
        char_weight = max(len(word), 1) / max(char_total, 1)
        duration = total_duration * char_weight
        timings.append(WordTiming(word=word, start_sec=round(cursor, 3), end_sec=round(cursor + duration, 3)))
        cursor += duration
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
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
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
                pcm = np.asarray(frame.to_ndarray())
                if pcm.ndim == 2:
                    if pcm.shape[0] == 1:
                        pcm = pcm[0]
                    elif pcm.shape[1] == 1:
                        pcm = pcm[:, 0]
                    else:
                        pcm = pcm.reshape(-1)
                pcm = pcm.astype(np.float32).reshape(-1)
                if pcm.size and np.max(np.abs(pcm)) <= 1.5:
                    pcm = np.clip(pcm * 32767.0, -32768, 32767)
                samples.append(pcm.astype(np.int16))
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
    sample_cursor = 0
    for packet in audio_container.demux(audio_stream):
        for frame in packet.decode():
            pcm = np.asarray(frame.to_ndarray())
            if pcm.ndim == 1:
                pcm = pcm.reshape(1, -1)
            elif pcm.shape[0] != 1 and pcm.shape[1] == 1:
                pcm = pcm.T
            aframe = av.AudioFrame.from_ndarray(pcm.astype(np.int16), format="s16p", layout="mono")
            aframe.sample_rate = audio_stream.rate or SAMPLE_RATE
            aframe.pts = sample_cursor
            aframe.time_base = Fraction(1, audio_stream.rate or SAMPLE_RATE)
            sample_cursor += aframe.samples
            for encoded in out_audio.encode(aframe):
                output.mux(encoded)
    for encoded in out_audio.encode():
        output.mux(encoded)
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
    output.close()
    video_container.close()
    audio_container.close()


def write_video_with_audio(frames: list[np.ndarray], audio_path: Path | None, output_path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("No frames supplied for video export.")
    ensure_dir(output_path.parent)
    temp_video = output_path.with_suffix(".video-only.mp4")
    export_video(frames, temp_video, fps)
    try:
        if audio_path is None or not audio_path.exists():
            temp_video.replace(output_path)
            return
        mux_video_audio(temp_video, audio_path, output_path)
    finally:
        if temp_video.exists():
            try:
                temp_video.unlink()
            except Exception:
                pass


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
