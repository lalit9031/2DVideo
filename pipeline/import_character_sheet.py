from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

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
    timestamp_iso,
    write_json,
)
from pipeline.bootstrap import _write_silent_reference_clip


@dataclass(frozen=True)
class CropSpec:
    name: str
    box: tuple[int, int, int, int]


def _remove_bg(img: Image.Image, threshold: int = 35) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    bg = arr[0, 0, :3].astype(int)
    diff = np.max(np.abs(arr[:, :, :3].astype(int) - bg), axis=2)
    mask = diff > threshold
    arr[:, :, 3] = np.where(mask, arr[:, :, 3], 0)
    return Image.fromarray(arr, "RGBA")


def _crop_and_save(source: Image.Image, spec: CropSpec, target_dir: Path) -> None:
    crop = source.crop(spec.box)
    crop = _remove_bg(crop)
    ensure_dir(target_dir)
    crop.save(target_dir / f"{spec.name}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a full character sheet into the cast registry.")
    parser.add_argument("--source", required=True, help="Path to the character sheet image.")
    parser.add_argument("--character-id", required=True, help="Target character id.")
    parser.add_argument("--name", required=True, help="Display name.")
    parser.add_argument("--voice-id", help="Voice id; defaults to character id.")
    parser.add_argument("--reference-frequency", type=float, default=205.0, help="Synthetic voice frequency for bootstrap clip.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = Image.open(args.source).convert("RGBA")
    character_id = args.character_id
    voice_id = args.voice_id or character_id
    target_dir = ensure_dir(character_assets_dir(character_id))
    source.save(target_dir / "source_sheet.png")

    # Top row face variants and lower-sheet body parts from the imported sheet.
    crops = [
        CropSpec("face_neutral", (1485, 160, 1854, 497)),
        CropSpec("face_blink", (680, 160, 1049, 497)),
        CropSpec("face_smile", (1889, 160, 2258, 497)),
        CropSpec("face_surprised", (1082, 160, 1451, 497)),
        CropSpec("face_wide", (279, 160, 648, 497)),
        CropSpec("body", (518, 918, 681, 1212)),
        CropSpec("arm_l", (435, 964, 520, 1129)),
        CropSpec("arm_r", (678, 964, 763, 1130)),
        CropSpec("leg_l", (509, 1238, 585, 1415)),
        CropSpec("leg_r", (613, 1238, 689, 1415)),
        CropSpec("shoe_l", (459, 1501, 553, 1589)),
        CropSpec("shoe_r", (645, 1501, 739, 1589)),
    ]
    for spec in crops:
        _crop_and_save(source, spec, target_dir)

    character_data = {
        "character_id": character_id,
        "name": args.name,
        "voice_id": voice_id,
        "rig_parts": ["face_neutral", "face_blink", "face_smile", "face_surprised", "face_wide", "body", "arm_l", "arm_r", "leg_l", "leg_r", "shoe_l", "shoe_r"],
        "anim_cycles": ["idle", "bounce_wave", "walk", "point", "jump", "sleep"],
        "pivot_points": {"head": [0, -120], "arm_l": [-60, -20], "arm_r": [60, -20]},
        "asset_dir": f"config/characters/{character_id}",
        "render_mode": "face_variants",
        "face_variants": {
            "eyes_open": "face_neutral",
            "eyes_blink": "face_blink",
            "mouth_closed": "face_smile",
            "mouth_open": "face_surprised",
            "mouth_wide": "face_wide",
            "head": "face_neutral",
        },
    }
    character_schema_validate(character_data)
    write_json(character_json_path(character_id), character_data)

    registry = load_registry()
    registry[character_id] = {
        "tier": "cast",
        "name": args.name,
        "voice_id": voice_id,
        "character_json": f"config/characters/{character_id}.json",
        "created": timestamp_iso(),
        "source": Path(args.source).name,
    }
    save_registry(registry)

    voices = load_voice_registry()
    ref_clip = CONFIG_DIR / "voices" / f"{voice_id}_ref.wav"
    _write_silent_reference_clip(ref_clip, args.reference_frequency)
    voices[voice_id] = {
        "reference_clip": f"config/voices/{voice_id}_ref.wav",
        "language": "en",
        "pitch_shift": 0,
        "speed": 1.0,
    }
    save_voice_registry(voices)

    print(character_id)


if __name__ == "__main__":
    main()
