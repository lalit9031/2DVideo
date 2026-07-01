from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline.common import character_assets_dir, load_registry, read_json
from pipeline.backgrounds import generate_background, background_path


FPS = 24.0


@dataclass(frozen=True)
class ShotRenderResult:
    shot_id: str
    video_path: Path
    frame_count: int
    fps: float


def _load_part(character_id: str, part: str) -> Image.Image:
    path = character_assets_dir(character_id) / f"{part}.png"
    print(f"[rig-load] {character_id}:{part} -> {path}", flush=True)
    if not path.exists():
        raise FileNotFoundError(f"Missing character part: {path}")
    return Image.open(path).convert("RGBA")


def _load_character(character_id: str) -> dict:
    registry = load_registry()
    entry = registry.get(character_id)
    if not entry:
        raise KeyError(f"Unknown character id: {character_id}")
    path = Path(entry["character_json"])
    if not path.is_absolute():
        from pipeline.common import ROOT

        path = ROOT / path
    return read_json(path)


def _background(size: tuple[int, int], background: str, time_factor: float) -> Image.Image:
    scene_id = background or "meadow_day"
    path = generate_background(scene_id, size)
    img = Image.open(path).convert("RGBA")
    width, height = size
    scale = 1.03 + 0.02 * math.sin(time_factor * 0.6)
    pan = int(math.sin(time_factor * 0.3) * 30)
    zoom_w = int(width / scale)
    zoom_h = int(height / scale)
    left = max(0, min(img.width - zoom_w, (img.width - zoom_w) // 2 + pan))
    top = max(0, min(img.height - zoom_h, (img.height - zoom_h) // 2))
    return img.crop((left, top, left + zoom_w, top + zoom_h)).resize(size, Image.Resampling.LANCZOS)


def _mouth_for_time(t: float, timings: Sequence[dict]) -> str:
    if not timings:
        return "mouth_closed"
    for index, item in enumerate(timings):
        if item["start_sec"] <= t <= item["end_sec"]:
            word = item["word"].lower()
            if len(word) <= 3:
                return "mouth_closed"
            if word.endswith(("a", "e", "i", "o", "u", "y")) or len(word) > 7:
                return "mouth_wide"
            return "mouth_open"
    return "mouth_closed"


def _eye_state(t: float) -> str:
    return "eyes_blink" if int(t * 3) % 11 == 0 else "eyes_open"


def _compose_character(
    base: Image.Image,
    character_id: str,
    *,
    x: int,
    y: int,
    scale: float,
    time_sec: float,
    mouth_state: str,
    action: str,
) -> None:
    character = _load_character(character_id)
    body = _load_part(character_id, "body").resize((int(280 * scale), int(360 * scale)))
    arm_l = _load_part(character_id, "arm_l").resize((int(120 * scale), int(240 * scale)))
    arm_r = _load_part(character_id, "arm_r").resize((int(120 * scale), int(240 * scale)))
    leg_l = _load_part(character_id, "leg_l").resize((int(120 * scale), int(220 * scale)))
    leg_r = _load_part(character_id, "leg_r").resize((int(120 * scale), int(220 * scale)))
    if character.get("render_mode") == "face_variants":
        variant_map = character.get("face_variants", {})
        face_key = variant_map.get(mouth_state, variant_map.get("head", "face_neutral"))
        head = _load_part(character_id, face_key).resize((int(256 * scale), int(256 * scale)))
        eyes = mouth = None
    else:
        head = _load_part(character_id, "head").resize((int(256 * scale), int(256 * scale)))
        eyes = _load_part(character_id, _eye_state(time_sec)).resize((int(256 * scale), int(256 * scale)))
        mouth = _load_part(character_id, mouth_state).resize((int(256 * scale), int(256 * scale)))
    bob = int(math.sin(time_sec * 4.0) * 8 * scale)
    if action == "jump":
        bob += int(abs(math.sin(time_sec * 5.5)) * 20 * scale)
    if action == "sleep":
        bob += 4
    if action == "walk":
        bob += int(math.sin(time_sec * 7.5) * 6 * scale)
    anchor_y = y + bob
    base.alpha_composite(leg_l, (x - leg_l.width // 2 - int(55 * scale), anchor_y + 180))
    base.alpha_composite(leg_r, (x - leg_r.width // 2 + int(55 * scale), anchor_y + 180))
    base.alpha_composite(body, (x - body.width // 2, anchor_y - 90))
    arm_offset = int(math.sin(time_sec * 4.0) * 12 * scale)
    if action == "point":
        arm_offset += int(18 * scale)
    if action == "bounce_wave":
        arm_offset += int(math.sin(time_sec * 8.0) * 28 * scale)
    base.alpha_composite(arm_l.rotate(-20 + arm_offset * 0.3, resample=Image.Resampling.BICUBIC, expand=True), (x - 150, anchor_y - 10))
    base.alpha_composite(arm_r.rotate(20 - arm_offset * 0.3, resample=Image.Resampling.BICUBIC, expand=True), (x + 30, anchor_y - 10))
    base.alpha_composite(head, (x - head.width // 2, anchor_y - 240))
    if eyes is not None:
        base.alpha_composite(eyes, (x - eyes.width // 2, anchor_y - 240))
    if mouth is not None:
        base.alpha_composite(mouth, (x - mouth.width // 2, anchor_y - 240))


def render_shot(
    episode: dict,
    shot: dict,
    output_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float = FPS,
) -> ShotRenderResult:
    duration = float(shot.get("duration_sec", 4))
    frame_count = max(1, int(round(duration * fps)))
    frames: list[np.ndarray] = []
    registry = load_registry()
    characters = shot.get("characters", [])
    dialogue = shot.get("dialogue", [])
    active_character = dialogue[0]["character"] if dialogue else (characters[0] if characters else None)
    timing = dialogue[0].get("timings", []) if dialogue else []
    action = shot.get("action", "idle")
    for index in range(frame_count):
        time_sec = index / fps
        background = _background((width, height), shot.get("background", "meadow_day"), time_sec)
        canvas = background.copy()
        if characters:
            spread = width // max(len(characters) + 1, 2)
            for idx, character_id in enumerate(characters):
                x = spread * (idx + 1)
                y = int(height * 0.62)
                char_action = action if character_id == active_character else "idle"
                _compose_character(
                    canvas,
                    character_id,
                    x=x,
                    y=y,
                    scale=0.9,
                    time_sec=time_sec + idx * 0.2,
                    mouth_state=_mouth_for_time(time_sec, timing) if character_id == active_character else "mouth_closed",
                    action=char_action,
                )
        else:
            draw = ImageDraw.Draw(canvas)
            for idx in range(6):
                offset = int(math.sin(time_sec * 1.8 + idx) * 12)
                x = 120 + idx * 180 + offset
                y = int(height * 0.55 + math.cos(time_sec * 1.4 + idx) * 28)
                draw.ellipse((x, y, x + 64, y + 64), fill=(255, 255, 255, 110), outline=(255, 220, 170, 180), width=4)
            if "flower" in shot.get("video_prompt", "").lower():
                for idx in range(7):
                    x = 90 + idx * 160 + int(math.sin(time_sec * 2.0 + idx) * 18)
                    y = int(height * 0.72 + math.cos(time_sec * 1.7 + idx) * 12)
                    draw.ellipse((x, y, x + 26, y + 26), fill=(255, 146, 182, 220))
        frames.append(np.array(canvas.convert("RGB")))
    from pipeline.media import export_video

    export_video(frames, output_path, fps)
    return ShotRenderResult(shot_id=shot["shot_id"], video_path=output_path, frame_count=frame_count, fps=fps)


def render_broll(
    shot: dict,
    output_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float = FPS,
) -> ShotRenderResult:
    duration = float(shot.get("duration_sec", 4))
    frame_count = max(1, int(round(duration * fps)))
    frames: list[np.ndarray] = []
    prompt = shot.get("video_prompt", "broll").lower()
    for index in range(frame_count):
        t = index / fps
        scene = "park_sunny"
        if "night" in prompt or "moon" in prompt:
            scene = "bedroom_night"
        elif "classroom" in prompt or "school" in prompt:
            scene = "classroom"
        elif "flower" in prompt or "garden" in prompt:
            scene = "garden"
        canvas = _background((width, height), scene, t)
        draw = ImageDraw.Draw(canvas)
        cx = int(width * 0.5 + math.sin(t * 1.6) * 120)
        cy = int(height * 0.48 + math.cos(t * 1.3) * 44)
        radius = int(120 + 18 * math.sin(t * 2.6))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 255, 255, 82), outline=(255, 225, 168, 160), width=4)
        for i in range(8):
            hue = (i * 35 + index * 3) % 255
            x = int(100 + i * 140 + math.sin(t * 2 + i) * 22)
            y = int(520 + math.cos(t * 1.8 + i) * 14)
            draw.ellipse((x, y, x + 46, y + 46), fill=(255, 190 - hue // 3, 140 + hue // 4, 200))
        frames.append(np.array(canvas.convert("RGB")))
    from pipeline.media import export_video

    export_video(frames, output_path, fps)
    return ShotRenderResult(shot_id=shot["shot_id"], video_path=output_path, frame_count=frame_count, fps=fps)
