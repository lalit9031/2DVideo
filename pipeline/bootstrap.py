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


DEMO_CAST: tuple[DemoCharacter, ...] = (
    DemoCharacter("char_01_bunny", "Bibi the Bunny", "char_01_bunny", (245, 163, 179)),
    DemoCharacter("char_02_fox", "Fifi the Fox", "char_02_fox", (250, 168, 72)),
    DemoCharacter("char_03_bear", "Bobo the Bear", "char_03_bear", (165, 124, 92)),
    DemoCharacter("char_04_cat", "Coco the Cat", "char_04_cat", (164, 188, 250)),
    DemoCharacter("char_05_owl", "Ollie the Owl", "char_05_owl", (182, 153, 250)),
)


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


def _draw_part_image(size: tuple[int, int], color: tuple[int, int, int], part: str) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if part == "head":
        draw.ellipse((18, 10, size[0] - 18, size[1] - 18), fill=color + (255,), outline=(40, 40, 40, 255), width=4)
        draw.ellipse((35, 38, 45, 48), fill=(25, 25, 25, 255))
        draw.ellipse((size[0] - 45, 38, size[0] - 35, 48), fill=(25, 25, 25, 255))
    elif part == "eyes_open":
        draw.ellipse((33, 35, 43, 45), fill=(15, 15, 15, 255))
        draw.ellipse((size[0] - 43, 35, size[0] - 33, 45), fill=(15, 15, 15, 255))
    elif part == "eyes_blink":
        draw.line((32, 40, 44, 40), fill=(15, 15, 15, 255), width=4)
        draw.line((size[0] - 44, 40, size[0] - 32, 40), fill=(15, 15, 15, 255), width=4)
    elif part == "mouth_closed":
        draw.rounded_rectangle((44, 85, size[0] - 44, 95), radius=4, fill=(70, 30, 40, 255))
    elif part == "mouth_open":
        draw.ellipse((40, 78, size[0] - 40, 105), fill=(120, 40, 60, 255), outline=(70, 20, 30, 255))
    elif part == "mouth_wide":
        draw.rounded_rectangle((34, 76, size[0] - 34, 110), radius=15, fill=(130, 35, 55, 255), outline=(70, 20, 30, 255), width=3)
    elif part == "body":
        draw.rounded_rectangle((28, 8, size[0] - 28, size[1] - 8), radius=42, fill=color + (255,), outline=(40, 40, 40, 255), width=4)
    elif part == "arm_l":
        draw.rounded_rectangle((14, 18, size[0] - 18, size[1] - 18), radius=18, fill=color + (255,), outline=(40, 40, 40, 255), width=3)
    elif part == "arm_r":
        draw.rounded_rectangle((18, 18, size[0] - 14, size[1] - 18), radius=18, fill=color + (255,), outline=(40, 40, 40, 255), width=3)
    elif part == "leg_l":
        draw.rounded_rectangle((18, 14, size[0] - 18, size[1] - 14), radius=16, fill=color + (255,), outline=(40, 40, 40, 255), width=3)
    elif part == "leg_r":
        draw.rounded_rectangle((18, 14, size[0] - 18, size[1] - 14), radius=16, fill=color + (255,), outline=(40, 40, 40, 255), width=3)
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
        _draw_part_image(size, character.color, part).save(base_dir / f"{part}.png")
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

