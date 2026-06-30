from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
SCHEMAS_DIR = ROOT / "schemas"
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "output"
LOGS_DIR = ROOT / "logs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: str | os.PathLike[str] | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or ROOT) / path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=True)
        fh.write("\n")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def schema_path(name: str) -> Path:
    return SCHEMAS_DIR / f"{name}.schema.json"


def validate_json_schema(name: str, data: Any) -> None:
    schema = read_json(schema_path(name))
    jsonschema.Draft202012Validator(schema).validate(data)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "item"


def today_iso() -> str:
    return date.today().isoformat()


def timestamp_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_registry() -> dict[str, Any]:
    path = CONFIG_DIR / "characters" / "registry.json"
    if not path.exists():
        return {}
    return read_json(path)


def save_registry(data: dict[str, Any]) -> None:
    write_json(CONFIG_DIR / "characters" / "registry.json", data)


def load_voice_registry() -> dict[str, Any]:
    path = CONFIG_DIR / "voices" / "voice_registry.json"
    if not path.exists():
        return {}
    return read_json(path)


def save_voice_registry(data: dict[str, Any]) -> None:
    write_json(CONFIG_DIR / "voices" / "voice_registry.json", data)


def character_json_path(character_id: str) -> Path:
    return CONFIG_DIR / "characters" / f"{character_id}.json"


def character_assets_dir(character_id: str) -> Path:
    return CONFIG_DIR / "characters" / character_id


def episode_schema_validate(data: Any) -> None:
    validate_json_schema("episode", data)


def character_schema_validate(data: Any) -> None:
    validate_json_schema("character", data)


def registry_schema_validate(data: Any) -> None:
    validate_json_schema("registry", data)


def voice_registry_schema_validate(data: Any) -> None:
    validate_json_schema("voice_registry", data)

