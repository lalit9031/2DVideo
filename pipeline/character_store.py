from __future__ import annotations

from pathlib import Path

from pipeline.bootstrap import bootstrap_guest_character
from pipeline.common import (
    character_json_path,
    ensure_dir,
    load_registry,
    read_json,
    save_registry,
    slugify,
    timestamp_iso,
    validate_json_schema,
    write_json,
)


def load_character(character_id: str) -> dict:
    path = character_json_path(character_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing character file for {character_id}: {path}")
    data = read_json(path)
    validate_json_schema("character", data)
    return data


def ensure_character_exists(character_id: str, *, source_episode: str | None = None) -> dict:
    registry = load_registry()
    if character_id not in registry:
        bootstrap_guest_character(character_id, _friendly_name(character_id), source_episode=source_episode)
        registry = load_registry()
    entry = registry[character_id]
    load_character(character_id)
    return entry


def ensure_episode_characters(episode: dict) -> dict:
    source_episode = episode.get("episode_id")
    for character_id in episode.get("characters_used", []):
        ensure_character_exists(character_id, source_episode=source_episode)
    for shot in episode.get("shots", []):
        for dialogue in shot.get("dialogue", []):
            character_id = dialogue["character"]
            ensure_character_exists(character_id, source_episode=source_episode)
            dialogue["voice_id"] = load_registry()[character_id]["voice_id"]
    return episode


def _friendly_name(character_id: str) -> str:
    pretty = slugify(character_id).replace("-", " ").title()
    return pretty

