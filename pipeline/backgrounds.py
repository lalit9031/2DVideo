from __future__ import annotations

from math import cos, sin, pi
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.common import ASSETS_DIR, ensure_dir


BACKGROUND_DIR = ASSETS_DIR / "backgrounds"


def background_path(scene_id: str) -> Path:
    return BACKGROUND_DIR / f"{scene_id}.png"


def _sky_gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    width, height = size
    img = Image.new("RGBA", size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line((0, y, width, y), fill=color + (255,))
    return img


def _add_clouds(draw: ImageDraw.ImageDraw, width: int, height: int, seed: int) -> None:
    for idx in range(5):
        cx = int(width * (0.12 + idx * 0.18) + (seed % 17) * 2)
        cy = int(height * 0.14 + (idx % 2) * 18)
        r = 42 + (idx % 3) * 8
        color = (255, 255, 255, 190)
        draw.ellipse((cx - r, cy - r * 0.55, cx + r, cy + r * 0.55), fill=color)
        draw.ellipse((cx - r * 1.1, cy - r * 0.35, cx + r * 0.2, cy + r * 0.7), fill=color)


def _add_sun(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    draw.ellipse((width - 220, 40, width - 100, 160), fill=(255, 224, 102, 255))


def _add_meadow(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    ground_y = int(height * 0.65)
    draw.rectangle((0, ground_y, width, height), fill=(120, 198, 109, 255))
    for idx in range(16):
        x = int(idx * width / 15)
        draw.line((x, ground_y, x + 20, height - 10), fill=(94, 168, 84, 255), width=3)
        draw.ellipse((x - 6, ground_y + 10, x + 6, ground_y + 22), fill=(245, 224, 120, 255))


def _add_classroom(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    desk_y = int(height * 0.66)
    draw.rectangle((0, desk_y, width, height), fill=(235, 211, 181, 255))
    for idx in range(6):
        x = 100 + idx * 180
        draw.rounded_rectangle((x, 250, x + 110, 380), radius=18, fill=(255, 244, 214, 255), outline=(177, 143, 82, 255), width=4)
    draw.rectangle((0, 0, width, 160), fill=(175, 229, 245, 110))


def _add_bedroom(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    draw.rectangle((0, int(height * 0.58), width, height), fill=(224, 206, 244, 255))
    draw.rounded_rectangle((120, 260, 430, 520), radius=28, fill=(255, 246, 232, 255), outline=(188, 155, 119, 255), width=5)
    draw.rounded_rectangle((140, 290, 410, 420), radius=20, fill=(246, 196, 206, 255))
    draw.ellipse((width - 220, 50, width - 80, 190), fill=(250, 240, 170, 255))


def _add_park(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    ground_y = int(height * 0.66)
    draw.rectangle((0, ground_y, width, height), fill=(107, 191, 92, 255))
    for x in range(60, width, 220):
        draw.rectangle((x, 220, x + 18, ground_y), fill=(121, 84, 45, 255))
        draw.ellipse((x - 70, 120, x + 90, 280), fill=(84, 165, 88, 255))
        draw.ellipse((x - 52, 94, x + 72, 228), fill=(93, 176, 95, 255))


def _add_rainbow(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    center = (width // 2, int(height * 0.7))
    colors = [(255, 115, 115, 255), (255, 182, 86, 255), (255, 244, 120, 255), (106, 209, 125, 255), (111, 192, 255, 255), (155, 140, 255, 255)]
    for idx, color in enumerate(colors):
        bbox = (center[0] - 300 + idx * 18, center[1] - 220 + idx * 18, center[0] + 300 - idx * 18, center[1] + 220 - idx * 18)
        draw.arc(bbox, start=180, end=360, fill=color, width=28)


def generate_background(scene_id: str, size: tuple[int, int] = (1280, 720)) -> Path:
    ensure_dir(BACKGROUND_DIR)
    path = background_path(scene_id)
    if path.exists():
        return path
    top, bottom = (180, 220, 255), (245, 250, 255)
    if scene_id in {"meadow_day", "garden"}:
        top, bottom = (187, 243, 204), (246, 252, 238)
    elif scene_id in {"bedroom_night"}:
        top, bottom = (102, 116, 181), (23, 26, 52)
    elif scene_id in {"classroom"}:
        top, bottom = (250, 237, 202), (255, 251, 243)
    elif scene_id in {"park_sunny"}:
        top, bottom = (170, 228, 255), (245, 252, 255)
    elif scene_id in {"rainbow_yard"}:
        top, bottom = (233, 227, 255), (253, 245, 255)
    img = _sky_gradient(size, top, bottom)
    draw = ImageDraw.Draw(img)
    width, height = size
    _add_sun(draw, width, height)
    _add_clouds(draw, width, height, seed=sum(map(ord, scene_id)))
    if scene_id in {"meadow_day", "garden"}:
        _add_meadow(draw, width, height)
        for x in range(120, width, 180):
            draw.ellipse((x, height - 120, x + 24, height - 30), fill=(255, 132, 170, 255))
    elif scene_id == "classroom":
        _add_classroom(draw, width, height)
    elif scene_id == "bedroom_night":
        _add_bedroom(draw, width, height)
    elif scene_id == "park_sunny":
        _add_park(draw, width, height)
    elif scene_id == "rainbow_yard":
        _add_rainbow(draw, width, height)
        draw.rectangle((0, int(height * 0.68), width, height), fill=(120, 205, 122, 255))
    else:
        _add_meadow(draw, width, height)
    img.save(path)
    return path

