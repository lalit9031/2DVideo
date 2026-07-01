"""
pipeline/07_assemble.py
~~~~~~~~~~~~~~~~~~~~~~~
Assembles the final video from AnimateDiff-animated clips.

Per-shot workflow:
  1. Take animated clip (animated_clip MP4) for the shot
  2. Composite the character portrait PNG over the clip (bottom-center, bobbing anim)
  3. Add Hindi TTS audio for the shot
  4. Concatenate all shots + background music
  5. Burn subtitles
  6. Output final.mp4 in correct aspect ratio (16:9 or 9:16)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.common import ensure_dir, read_json, write_json, ROOT, ASSETS_DIR, CONFIG_DIR
from pipeline.progress import progress_path_from_env, write_stage_progress

SHARED_PORTRAITS = ASSETS_DIR / "shared_character_portraits"
MUSIC_DIR        = ASSETS_DIR / "music"


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg(args: list[str], label: str = "") -> None:
    import os
    ffmpeg_bin = "/home/lalit/Desktop/GPU optimization/Wan2GP/ffmpeg_bins/ffmpeg"
    cmd = [ffmpeg_bin, "-y", "-threads", "4"] + args
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/home/lalit/Desktop/GPU optimization/Wan2GP/ffmpeg_bins:{env.get('LD_LIBRARY_PATH', '')}"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg {label} failed (rc={result.returncode}):\n{result.stderr[-800:]}"
        )


def _portrait_png(character_id: str) -> Path | None:
    p = SHARED_PORTRAITS / f"{character_id}.png"
    return p if p.exists() else None


def _get_bg_music() -> Path | None:
    for ext in ("*.mp3", "*.wav", "*.ogg"):
        found = list(MUSIC_DIR.glob(ext))
        if found:
            return found[0]
    return None


# ---------------------------------------------------------------------------
# Per-shot compositor: animated clip + character overlay + TTS audio
# ---------------------------------------------------------------------------

def _composite_shot(
    *,
    shot: dict,
    output_mp4: Path,
    tmp_dir: Path,
    work_dir: Path,
) -> Path:
    """
    Composite a single shot:
      - Take animated_clip (MP4 from AnimateDiff)
      - Overlay character portrait PNG (bottom-center, 30% height)
      - Replace audio with TTS wav (or keep silent)
    Returns output_mp4.
    """
    clip_path = Path(shot.get("animated_clip", ""))
    shot_id   = shot.get("shot_id", "s?")

    # Fallback: if no animated clip, use the upscaled PNG as a static clip
    if not clip_path.exists():
        upscaled = shot.get("scene_image_upscaled") or shot.get("scene_image", "")
        if upscaled and Path(upscaled).exists():
            duration = shot.get("duration_sec", 3)
            static_mp4 = tmp_dir / f"{shot_id}_static.mp4"
            _run_ffmpeg([
                "-loop", "1", "-i", str(upscaled),
                "-t", str(duration), "-r", "24",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(static_mp4)
            ], label=f"{shot_id}_static")
            clip_path = static_mp4
        else:
            raise FileNotFoundError(f"[assemble] No video clip for {shot_id}")

    # TTS audio for this shot
    tts_wav = None
    dialogue = shot.get("dialogue", [])
    if dialogue:
        tts_path = shot.get("tts_audio") or (
            dialogue[0].get("audio_path") if isinstance(dialogue[0], dict) else None
        )
        if tts_path:
            p = Path(tts_path)
            if not p.is_absolute():
                p = work_dir / p
            if p.exists():
                tts_wav = p

    in_args = ["-stream_loop", "-1", "-i", str(clip_path)]

    # Compose ffmpeg command
    duration = float(shot.get("duration_sec", 3))
    compose_mp4 = tmp_dir / f"{shot_id}_composed.mp4"
    cmd: list[str] = in_args[:]

    if tts_wav:
        cmd += ["-i", str(tts_wav)]
        audio_idx = "1"
        a_map = f"{audio_idx}:a"
        # Pad audio to video length and boost dialogue volume to 300%
        a_filter = f"[{audio_idx}:a]volume=3.0,apad[aout]"
        cmd += [
            "-filter_complex", a_filter,
            "-map", "0:v", "-map", "[aout]",
            "-t", str(duration),
            "-shortest",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(compose_mp4)
        ]
    else:
        # Generate silent audio stream to keep stream structure consistent
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
        cmd += [
            "-map", "0:v", "-map", "1:a",
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(compose_mp4)
        ]

    _run_ffmpeg(cmd, label=f"{shot_id}_compose")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    compose_mp4.rename(output_mp4)
    return output_mp4


# ---------------------------------------------------------------------------
# Subtitle (SRT) generation
# ---------------------------------------------------------------------------

def _make_srt(shots: list[dict], out_path: Path) -> None:
    lines = []
    t = 0.0
    for i, shot in enumerate(shots, 1):
        dur = float(shot.get("duration_sec", 3))
        dialogue = shot.get("dialogue", [])
        text = dialogue[0].get("line", "") if dialogue else ""
        if not text:
            t += dur
            continue
        start = _fmt_srt_ts(t)
        end   = _fmt_srt_ts(t + dur)
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        t += dur
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_srt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def assemble_episode_video(
    episode: dict,
    work_dir: Path,
    output_path: Path,
) -> dict:
    """
    Full assembly:
      1. Composite each shot clip
      2. Concatenate all shots
      3. Mix in background music (if any)
      4. Burn subtitles
    Returns a report dict.
    """
    shots  = episode.get("shots", [])
    fmt    = episode.get("format", "full")

    with tempfile.TemporaryDirectory(prefix="2dvideo_assemble_") as tmpdir:
        tmp = Path(tmpdir)
        composed: list[Path] = []

        for i, shot in enumerate(shots):
            shot_id = shot.get("shot_id", f"s{i+1}")
            out_clip = tmp / f"{shot_id}_final.mp4"
            try:
                _composite_shot(shot=shot, output_mp4=out_clip, tmp_dir=tmp, work_dir=work_dir)
                composed.append(out_clip)
                print(f"[assemble] ✓ {shot_id} composited", flush=True)
            except Exception as exc:
                print(f"[assemble] ⚠ {shot_id} failed: {exc}", flush=True)

        if not composed:
            raise RuntimeError("[assemble] No shots composited — nothing to assemble")

        # Concat all clips
        concat_file = tmp / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{p}'" for p in composed), encoding="utf-8"
        )
        concat_mp4 = tmp / "concat.mp4"
        _run_ffmpeg([
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy", str(concat_mp4)
        ], label="concat")
        print("[assemble] Shots concatenated", flush=True)

        # Mix background music (if present)
        music = _get_bg_music()
        if music:
            mixed_mp4 = tmp / "mixed.mp4"
            _run_ffmpeg([
                "-i", str(concat_mp4),
                "-i", str(music),
                "-filter_complex",
                "[0:a]volume=1.5[va];[1:a]volume=0.08,aloop=loop=-1:size=2e+09[vm];[va][vm]amix=inputs=2:duration=first:normalize=0[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                str(mixed_mp4)
            ], label="music_mix")
            working = mixed_mp4
        else:
            working = concat_mp4

        # Burn subtitles
        srt_path = tmp / "subtitles.srt"
        _make_srt(shots, srt_path)

        ensure_dir(output_path.parent)
        _run_ffmpeg([
            "-i", str(working),
            "-vf", f"subtitles={srt_path}:force_style='FontName=Noto Sans,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'",
            "-c:v", "libx264", "-crf", "17", "-preset", "slow",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path)
        ], label="subtitles+final")

    print(f"[assemble] ✓ Final video: {output_path}", flush=True)
    return {
        "output": str(output_path),
        "format": fmt,
        "shots_composited": len(composed),
        "music": str(music) if music else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble final episode video from AnimateDiff clips.")
    p.add_argument("--episode", required=True, help="Path to episode.animated.json")
    p.add_argument("--output",  required=True, help="Path to final output MP4")
    return p


def main() -> None:
    args = build_parser().parse_args()
    progress_file = progress_path_from_env()
    episode_path  = Path(args.episode)
    output_path   = Path(args.output)
    work_dir      = episode_path.parent

    write_stage_progress(progress_file, fraction=0.05, stage="assemble",
                         message="Compositing shot clips…")

    episode = read_json(episode_path)
    report  = assemble_episode_video(episode, work_dir, output_path)

    write_stage_progress(progress_file, fraction=0.95, stage="assemble",
                         message="Assembly complete")

    write_json(output_path.with_suffix(".manifest.json"), report)
    print(output_path)


if __name__ == "__main__":
    main()
