from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline.common import character_assets_dir, load_registry, read_json


FPS = 12.0


@dataclass(frozen=True)
class ShotRenderResult:
    shot_id: str
    video_path: Path
    frame_count: int
    fps: float


def _load_part(character_id: str, part: str) -> Image.Image:
    return Image.open(character_assets_dir(character_id) / f"{part}.png").convert("RGBA")


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
    width, height = size
    img = Image.new("RGBA", size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    top = (180, 220, 255)
    bottom = (245, 250, 255)
    if "meadow" in background or "garden" in background:
        top, bottom = (185, 240, 192), (245, 252, 234)
    elif "classroom" in background or "room" in background:
        top, bottom = (253, 235, 208), (255, 248, 239)
    elif "rainbow" in background:
        top, bottom = (219, 218, 255), (250, 235, 255)
    for y in range(height):
        t = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line((0, y, width, y), fill=color + (255,))
    accent_x = int((width * 0.5) + math.sin(time_factor) * width * 0.06)
    draw.ellipse((accent_x - 90, 50, accent_x + 90, 230), fill=(255, 255, 255, 80))
    draw.ellipse((accent_x + 120, 68, accent_x + 190, 138), fill=(255, 255, 255, 60))
    return img


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
            prompt = shot.get("video_prompt", "broll")
            draw = ImageDraw.Draw(canvas)
            draw.rounded_rectangle((120, 120, width - 120, height - 120), radius=50, fill=(255, 255, 255, 65), outline=(255, 255, 255, 140), width=4)
            draw.text((170, 160), prompt[:120], fill=(50, 50, 50, 255))
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
    prompt = shot.get("video_prompt", "broll")
    for index in range(frame_count):
        t = index / fps
        canvas = _background((width, height), "rainbow", t)
        draw = ImageDraw.Draw(canvas)
        cx = int(width * 0.5 + math.sin(t * 2.0) * 90)
        cy = int(height * 0.5 + math.cos(t * 1.7) * 40)
        radius = int(130 + 25 * math.sin(t * 3.0))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 255, 255, 95), outline=(255, 220, 255, 180), width=5)
        draw.text((110, 110), prompt, fill=(50, 50, 50, 255))
        for i in range(7):
            hue = (i * 40 + index * 5) % 255
            x = int(120 + i * 150 + math.sin(t * 3 + i) * 18)
            y = int(520 + math.cos(t * 2 + i) * 20)
            draw.rounded_rectangle((x, y, x + 88, y + 88), radius=18, fill=(255, 200 - hue // 2, 150 + hue // 3, 210))
        frames.append(np.array(canvas.convert("RGB")))
    from pipeline.media import export_video

    export_video(frames, output_path, fps)
    return ShotRenderResult(shot_id=shot["shot_id"], video_path=output_path, frame_count=frame_count, fps=fps)
