from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import wave

import cv2
import numpy as np
import av

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ROOT, character_assets_dir, read_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run automated QA checks for a rendered episode.")
    parser.add_argument("--episode", required=True, help="Path to the episode JSON used for the render.")
    parser.add_argument("--input", required=True, help="Path to the final MP4 to validate.")
    parser.add_argument("--output", help="Optional QA report path.")
    parser.add_argument("--silence-threshold", type=float, default=5.0, help="Max allowed silent gap in seconds (default: 5.0; B-roll shots have no voice).")
    parser.add_argument("--duration-tolerance", type=float, default=1.0, help="Allowed duration mismatch in seconds.")
    return parser


def _seconds_from_stream(stream: av.stream.Stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if stream.start_time is not None and stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    return None


def _decode_audio_samples(path: Path) -> tuple[np.ndarray, int]:
    container = av.open(str(path))
    audio_stream = next((s for s in container.streams.audio), None)
    if audio_stream is None:
        container.close()
        return np.zeros(0, dtype=np.float32), 0
    sample_rate = int(audio_stream.rate or 22050)
    samples: list[np.ndarray] = []
    try:
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                arr = np.asarray(frame.to_ndarray())
                if arr.ndim == 2:
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.shape[1] == 1:
                        arr = arr[:, 0]
                    else:
                        arr = arr.reshape(-1)
                samples.append(arr.astype(np.float32).reshape(-1))
    finally:
        container.close()
    if not samples:
        return np.zeros(0, dtype=np.float32), sample_rate
    merged = np.concatenate(samples)
    return merged, sample_rate


def _longest_silence(samples: np.ndarray, sample_rate: int, window_sec: float = 0.1) -> float:
    if samples.size == 0 or sample_rate <= 0:
        return 0.0
    window = max(1, int(round(sample_rate * window_sec)))
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    threshold = max(0.005, peak * 0.015) if peak <= 2.0 else max(500.0, peak * 0.015)
    silent_windows = 0
    longest = 0
    for start in range(0, len(samples), window):
        block = samples[start:start + window]
        if block.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(block.astype(np.float32)))))
        if rms < threshold:
            silent_windows += 1
            longest = max(longest, silent_windows)
        else:
            silent_windows = 0
    return longest * window / sample_rate


def _sample_video_frames(path: Path, sample_count: int = 16) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return [], 0.0
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else 0.0
    sample_times = np.linspace(0.0, max(0.0, duration - 1.0 / fps), num=max(2, sample_count))
    frames: list[np.ndarray] = []
    try:
        for time_sec in sample_times:
            capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        capture.release()
    return frames, fps


def _frame_stats(frames: list[np.ndarray]) -> dict[str, float]:
    if not frames:
        return {"sampled_frames": 0.0, "black_frame_ratio": 1.0, "max_mean_diff": 0.0}
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    means = [float(gray.mean()) for gray in grays]
    black_ratio = sum(mean < 8.0 for mean in means) / len(means)
    diffs = []
    for prev, cur in zip(grays, grays[1:]):
        diff = float(np.mean(cv2.absdiff(prev, cur)))
        diffs.append(diff)
    return {
        "sampled_frames": float(len(frames)),
        "black_frame_ratio": float(black_ratio),
        "max_mean_diff": float(max(diffs) if diffs else 0.0),
    }


def _write_silence(path: Path, duration_sec: float, sample_rate: int = 22050) -> None:
    samples = np.zeros(max(1, int(round(duration_sec * sample_rate))), dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(samples.tobytes())


def _asset_coverage(episode: dict) -> dict[str, object]:
    from pipeline.common import CONFIG_DIR, ROOT
    used = list(episode.get("characters_used", []))
    shot_characters = {
        character
        for shot in episode.get("shots", [])
        if not shot.get("broll")
        for character in shot.get("characters", [])
    }
    asset_status: dict[str, bool] = {}
    for character_id in used:
        # Check if portrait mode
        char_json = CONFIG_DIR / "characters" / f"{character_id}.json"
        is_portrait = False
        if char_json.exists():
            try:
                char_data = read_json(char_json)
                if char_data.get("render_mode") == "portrait":
                    is_portrait = True
            except Exception:
                pass
        
        if is_portrait:
            shared_p = ROOT / "assets" / "shared_character_portraits" / f"{character_id}.png"
            local_p = character_assets_dir(character_id) / "portrait.png"
            asset_status[character_id] = shared_p.exists() or local_p.exists()
        else:
            asset_status[character_id] = character_assets_dir(character_id).exists()
            
    return {
        "used": used,
        "appears_in_shots": sorted(shot_characters),
        "assets_found": asset_status,
        "pass": all(character in shot_characters for character in used) and all(asset_status.values()),
    }


def main() -> None:
    args = build_parser().parse_args()
    episode_path = Path(args.episode)
    video_path = Path(args.input)
    report_path = Path(args.output) if args.output else video_path.with_suffix(".qa_report.json")
    episode = read_json(episode_path)

    checks: dict[str, dict[str, object]] = {}
    passed = True

    try:
        container = av.open(str(video_path))
    except Exception as exc:
        checks["open_video"] = {"pass": False, "error": str(exc)}
        report = {"episode_id": episode.get("episode_id"), "passed": False, "checks": checks}
        write_json(report_path, report)
        raise SystemExit(1)

    with container:
        video_stream = next((s for s in container.streams.video), None)
        audio_stream = next((s for s in container.streams.audio), None)
        streams_present = {"video": video_stream is not None, "audio": audio_stream is not None}
        checks["streams_present"] = {"pass": all(streams_present.values()), **streams_present}
        passed &= checks["streams_present"]["pass"]

        video_sec = _seconds_from_stream(video_stream) if video_stream else None
        audio_sec = _seconds_from_stream(audio_stream) if audio_stream else None
        expected_sec = float(episode.get("target_duration_sec", 0))
        duration_pass = (
            video_sec is not None
            and audio_sec is not None
            and abs(video_sec - expected_sec) <= args.duration_tolerance
            and abs(audio_sec - expected_sec) <= args.duration_tolerance
        )
        checks["duration_match"] = {
            "pass": duration_pass,
            "video_sec": round(video_sec, 3) if video_sec is not None else None,
            "audio_sec": round(audio_sec, 3) if audio_sec is not None else None,
            "expected_sec": expected_sec,
            "tolerance_sec": args.duration_tolerance,
        }
        passed &= duration_pass

        fps_value = float(video_stream.average_rate) if video_stream and video_stream.average_rate else 0.0
        frame_rate_pass = fps_value >= 24.0
        checks["frame_rate"] = {"pass": frame_rate_pass, "value": round(fps_value, 3)}
        passed &= frame_rate_pass

        codec_pass = bool(video_stream and audio_stream and video_stream.codec.name == "h264" and audio_stream.codec.name == "aac")
        checks["codec"] = {
            "pass": codec_pass,
            "video_codec": video_stream.codec.name if video_stream else None,
            "audio_codec": audio_stream.codec.name if audio_stream else None,
        }
        passed &= codec_pass

    samples, sample_rate = _decode_audio_samples(video_path)
    longest_gap = _longest_silence(samples, sample_rate)
    silence_pass = longest_gap <= args.silence_threshold
    checks["silence_gaps"] = {
        "pass": silence_pass,
        "longest_gap_sec": round(longest_gap, 3),
        "threshold_sec": args.silence_threshold,
        "sample_rate": sample_rate,
    }
    passed &= silence_pass

    frames, fps_value = _sample_video_frames(video_path)
    frame_stats = _frame_stats(frames)
    frozen_pass = frame_stats["max_mean_diff"] >= 1.5 and frame_stats["black_frame_ratio"] < 0.5
    checks["frozen_frames"] = {"pass": frozen_pass, **frame_stats, "fps": round(fps_value, 3)}
    passed &= frozen_pass

    file_size = video_path.stat().st_size if video_path.exists() else 0
    bytes_per_sec = file_size / max(expected_sec, 1.0)
    size_pass = 5_000 <= bytes_per_sec <= 2_000_000
    checks["file_size"] = {
        "pass": size_pass,
        "bytes": file_size,
        "bytes_per_sec": round(bytes_per_sec, 2),
    }
    passed &= size_pass

    asset_check = _asset_coverage(episode)
    checks["asset_coverage"] = asset_check
    passed &= bool(asset_check["pass"])

    report = {
        "episode_id": episode.get("episode_id"),
        "video_path": str(video_path),
        "passed": passed,
        "checks": checks,
    }
    write_json(report_path, report)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
