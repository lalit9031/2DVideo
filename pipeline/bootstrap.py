from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from pipeline.common import (
    CONFIG_DIR,
    character_assets_dir,
    character_json_path,
    character_schema_validate,
    ensure_dir,
    load_registry,
    load_voice_registry,
    save_registry,
    save_voice_registry,
    slugify,
    timestamp_iso,
    write_json,
)
from pipeline.media import SAMPLE_RATE
import wave
import numpy as np


@dataclass(frozen=True)
class DemoCharacter:
    character_id: str
    name: str
    voice_id: str
    color: tuple[int, int, int]


# (color, eye_color, accent_color)
DemoCharacter2 = DemoCharacter  # alias kept for compat

DEMO_CAST: tuple[DemoCharacter, ...] = (
    DemoCharacter("char_01_bunny", "Bibi the Bunny", "char_01_bunny", (245, 163, 179)),   # warm pink
    DemoCharacter("char_02_fox",   "Fifi the Fox",   "char_02_fox",   (250, 168,  72)),   # tangerine orange
    DemoCharacter("char_03_bear",  "Bobo the Bear",  "char_03_bear",  (165, 124,  92)),   # warm brown
    DemoCharacter("char_04_cat",   "Coco the Cat",   "char_04_cat",   (164, 188, 250)),   # sky blue
    DemoCharacter("char_05_owl",   "Ollie the Owl",  "char_05_owl",   (182, 153, 250)),   # lavender
)

_ACCENT: dict[str, tuple[int, int, int]] = {
    "char_01_bunny": (255, 220, 230),   # light blush inner-ear
    "char_02_fox":   (230, 115,  40),   # darker orange muzzle
    "char_03_bear":  (210, 170, 130),   # lighter snout
    "char_04_cat":   (110, 145, 235),   # darker blue cheek
    "char_05_owl":   (130,  90, 220),   # deeper purple brow
}

_EYE_COLOR: dict[str, tuple[int, int, int]] = {
    "char_01_bunny": ( 40,  10,  40),
    "char_02_fox":   ( 80,  30,   0),
    "char_03_bear":  ( 50,  20,   0),
    "char_04_cat":   ( 20,  60, 120),
    "char_05_owl":   (255, 200,   0),
}


def _write_silent_reference_clip(path: Path, frequency: float) -> None:
    ensure_dir(path.parent)
    duration = 20.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave_data = 0.18 * np.sin(2 * np.pi * frequency * t) * (np.sin(np.linspace(0, np.pi, len(t))) ** 1.4)
    pcm = (np.clip(wave_data, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes(pcm.tobytes())


def _draw_part_image(
    size: tuple[int, int],
    color: tuple[int, int, int],
    part: str,
    character_id: str = "",
) -> Image.Image:
    """Draw a Cocomelon-style flat-art character part for the given character."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    accent = _ACCENT.get(character_id, (220, 220, 220))
    eye_col = _EYE_COLOR.get(character_id, (20, 20, 20))
    outline = (35, 25, 40, 255)
    fill4 = color + (255,)
    w, h = size

    if part == "head":
        # Base head shape
        draw.ellipse((14, 12, w - 14, h - 14), fill=fill4, outline=outline, width=4)
        # Character-specific head features
        if character_id == "char_01_bunny":
            # Long upright bunny ears
            draw.ellipse((42, -36, 72, 60), fill=fill4, outline=outline, width=3)
            draw.ellipse((w - 72, -36, w - 42, 60), fill=fill4, outline=outline, width=3)
            draw.ellipse((50, -28, 64, 52), fill=accent + (255,), outline=(0,0,0,0))
            draw.ellipse((w - 64, -28, w - 50, 52), fill=accent + (255,), outline=(0,0,0,0))
        elif character_id == "char_02_fox":
            # Pointy triangular fox ears
            draw.polygon([(38, 50), (20, -10), (68, 36)], fill=fill4, outline=outline)
            draw.polygon([(w - 38, 50), (w - 20, -10), (w - 68, 36)], fill=fill4, outline=outline)
            draw.polygon([(44, 44), (28, 2), (62, 32)], fill=accent + (255,))
            draw.polygon([(w - 44, 44), (w - 28, 2), (w - 62, 32)], fill=accent + (255,))
            # Fox muzzle
            draw.ellipse((w//2 - 30, h//2 + 4, w//2 + 30, h - 22), fill=accent + (255,), outline=outline, width=2)
        elif character_id == "char_03_bear":
            # Rounded bear ears
            draw.ellipse((22, 8, 62, 48), fill=fill4, outline=outline, width=3)
            draw.ellipse((w - 62, 8, w - 22, 48), fill=fill4, outline=outline, width=3)
            draw.ellipse((30, 16, 54, 40), fill=accent + (255,))
            draw.ellipse((w - 54, 16, w - 30, 40), fill=accent + (255,))
            # Bear snout
            draw.ellipse((w//2 - 36, h//2 + 8, w//2 + 36, h - 16), fill=accent + (255,), outline=outline, width=2)
        elif character_id == "char_04_cat":
            # Cat pointed ears
            draw.polygon([(28, 48), (36, 0), (74, 42)], fill=fill4, outline=outline)
            draw.polygon([(w - 28, 48), (w - 36, 0), (w - 74, 42)], fill=fill4, outline=outline)
            draw.polygon([(38, 42), (44, 10), (68, 38)], fill=accent + (200,))
            draw.polygon([(w - 38, 42), (w - 44, 10), (w - 68, 38)], fill=accent + (200,))
        elif character_id == "char_05_owl":
            # Owl ear tufts
            draw.polygon([(44, 28), (52, -8), (72, 30)], fill=fill4, outline=outline)
            draw.polygon([(w - 44, 28), (w - 52, -8), (w - 72, 30)], fill=fill4, outline=outline)
            # Owl facial disk
            draw.ellipse((22, 30, w - 22, h - 10), fill=accent + (80,))

    elif part == "eyes_open":
        r = 14  # eye radius
        lx, ly = w // 2 - 36, h // 2 - 14
        rx, ry = w // 2 + 36, h // 2 - 14
        if character_id == "char_05_owl":
            r = 22  # owls have large eyes
        # White sclera
        draw.ellipse((lx - r, ly - r, lx + r, ly + r), fill=(255, 255, 255, 255), outline=outline, width=3)
        draw.ellipse((rx - r, ry - r, rx + r, ry + r), fill=(255, 255, 255, 255), outline=outline, width=3)
        # Iris
        ir = max(6, r - 5)
        draw.ellipse((lx - ir, ly - ir, lx + ir, ly + ir), fill=eye_col + (255,))
        draw.ellipse((rx - ir, ry - ir, rx + ir, ry + ir), fill=eye_col + (255,))
        # Pupil highlight
        draw.ellipse((lx - 3, ly - 5, lx + 3, ly + 1), fill=(255, 255, 255, 200))
        draw.ellipse((rx - 3, ry - 5, rx + 3, ry + 1), fill=(255, 255, 255, 200))
        # Cat whiskers
        if character_id == "char_04_cat":
            wcy = h // 2 + 16
            draw.line((lx - 20, wcy, lx + 22, wcy - 3), fill=(50, 50, 60, 200), width=2)
            draw.line((lx - 20, wcy + 6, lx + 22, wcy + 3), fill=(50, 50, 60, 200), width=2)
            draw.line((rx + 20, wcy, rx - 22, wcy - 3), fill=(50, 50, 60, 200), width=2)
            draw.line((rx + 20, wcy + 6, rx - 22, wcy + 3), fill=(50, 50, 60, 200), width=2)

    elif part == "eyes_blink":
        lx, ly = w // 2 - 36, h // 2 - 14
        rx, ry = w // 2 + 36, h // 2 - 14
        bw = 14 if character_id == "char_05_owl" else 10
        draw.arc((lx - bw, ly - 6, lx + bw, ly + 10), start=200, end=340, fill=outline, width=4)
        draw.arc((rx - bw, ry - 6, rx + bw, ry + 10), start=200, end=340, fill=outline, width=4)

    elif part == "mouth_closed":
        mx, my = w // 2, h // 2 + 34
        draw.arc((mx - 18, my - 8, mx + 18, my + 10), start=10, end=170, fill=outline, width=4)

    elif part == "mouth_open":
        mx, my = w // 2, h // 2 + 32
        draw.ellipse((mx - 24, my - 10, mx + 24, my + 22), fill=(180, 45, 65, 255), outline=outline, width=3)
        # Teeth strip
        draw.rounded_rectangle((mx - 18, my - 6, mx + 18, my + 4), radius=3, fill=(255, 250, 248, 255))

    elif part == "mouth_wide":
        mx, my = w // 2, h // 2 + 30
        draw.ellipse((mx - 36, my - 14, mx + 36, my + 28), fill=(190, 40, 60, 255), outline=outline, width=3)
        draw.rounded_rectangle((mx - 28, my - 10, mx + 28, my + 4), radius=4, fill=(255, 250, 248, 255))
        draw.line((mx, my - 10, mx, my + 4), fill=(220, 190, 190, 200), width=2)

    elif part == "body":
        # Rounded torso with belly accent
        draw.rounded_rectangle((22, 10, w - 22, h - 10), radius=44, fill=fill4, outline=outline, width=4)
        draw.ellipse((w//2 - 32, h//2 - 10, w//2 + 32, h//2 + 50), fill=accent + (160,))
        # Neckline detail
        draw.arc((w//2 - 24, 4, w//2 + 24, 28), start=30, end=150, fill=outline, width=3)

    elif part == "arm_l":
        draw.rounded_rectangle((10, 12, w - 10, h - 18), radius=20, fill=fill4, outline=outline, width=3)
        # Hand
        draw.ellipse((10, h - 42, w - 10, h - 8), fill=fill4, outline=outline, width=2)

    elif part == "arm_r":
        draw.rounded_rectangle((10, 12, w - 10, h - 18), radius=20, fill=fill4, outline=outline, width=3)
        draw.ellipse((10, h - 42, w - 10, h - 8), fill=fill4, outline=outline, width=2)

    elif part == "leg_l":
        draw.rounded_rectangle((16, 10, w - 16, h - 22), radius=18, fill=fill4, outline=outline, width=3)
        # Foot / shoe
        draw.ellipse((6, h - 44, w - 4, h - 8), fill=accent + (255,), outline=outline, width=2)

    elif part == "leg_r":
        draw.rounded_rectangle((16, 10, w - 16, h - 22), radius=18, fill=fill4, outline=outline, width=3)
        draw.ellipse((4, h - 44, w - 6, h - 8), fill=accent + (255,), outline=outline, width=2)

    return img


def _write_demo_character_assets(character: DemoCharacter) -> None:
    base_dir = ensure_dir(character_assets_dir(character.character_id))
    specs = {
        "head": (256, 256),
        "eyes_open": (256, 256),
        "eyes_blink": (256, 256),
        "mouth_closed": (256, 256),
        "mouth_open": (256, 256),
        "mouth_wide": (256, 256),
        "body": (280, 360),
        "arm_l": (120, 240),
        "arm_r": (120, 240),
        "leg_l": (120, 220),
        "leg_r": (120, 220),
    }
    for part, size in specs.items():
        _draw_part_image(size, character.color, part, character.character_id).save(base_dir / f"{part}.png")
    character_data = {
        "character_id": character.character_id,
        "name": character.name,
        "voice_id": character.voice_id,
        "rig_parts": [
            "head",
            "eyes_open",
            "eyes_blink",
            "mouth_closed",
            "mouth_open",
            "mouth_wide",
            "body",
            "arm_l",
            "arm_r",
            "leg_l",
            "leg_r",
        ],
        "anim_cycles": ["idle", "bounce_wave", "walk", "point", "jump", "sleep"],
        "pivot_points": {"head": [0, -120], "arm_l": [-60, -20], "arm_r": [60, -20]},
        "asset_dir": f"config/characters/{character.character_id}",
    }
    character_schema_validate(character_data)
    write_json(character_json_path(character.character_id), character_data)


def bootstrap_demo_cast(force: bool = False) -> None:
    registry = load_registry()
    voices = load_voice_registry()
    updated = False
    ensure_dir(CONFIG_DIR / "characters")
    ensure_dir(CONFIG_DIR / "voices")
    for index, character in enumerate(DEMO_CAST):
        if force or character.character_id not in registry:
            _write_demo_character_assets(character)
            registry[character.character_id] = {
                "tier": "cast",
                "name": character.name,
                "voice_id": character.voice_id,
                "character_json": f"config/characters/{character.character_id}.json",
                "created": timestamp_iso(),
            }
            updated = True
        if force or character.voice_id not in voices:
            ref_clip = CONFIG_DIR / "voices" / f"{character.voice_id}_ref.wav"
            _write_silent_reference_clip(ref_clip, 170 + index * 30)
            voices[character.voice_id] = {
                "reference_clip": f"config/voices/{character.voice_id}_ref.wav",
                "language": "en",
                "pitch_shift": 0,
                "speed": 1.0,
            }
            updated = True
    if updated:
        save_registry(registry)
        save_voice_registry(voices)


def bootstrap_guest_character(
    character_id: str,
    name: str,
    color: tuple[int, int, int] = (190, 190, 190),
    source_episode: str | None = None,
) -> None:
    character = DemoCharacter(character_id, name, character_id, color)
    _write_demo_character_assets(character)
    registry = load_registry()
    registry[character_id] = {
        "tier": "guest",
        "name": name,
        "voice_id": character_id,
        "character_json": f"config/characters/{character_id}.json",
        "created": timestamp_iso(),
        **({"source_episode": source_episode} if source_episode else {}),
    }
    voices = load_voice_registry()
    ref_clip = CONFIG_DIR / "voices" / f"{character_id}_ref.wav"
    _write_silent_reference_clip(ref_clip, 210)
    voices[character_id] = {
        "reference_clip": f"config/voices/{character_id}_ref.wav",
        "language": "en",
        "pitch_shift": 0,
        "speed": 1.0,
    }
    save_registry(registry)
    save_voice_registry(voices)

