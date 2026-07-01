#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIBLE_DIR = ROOT / "prompt_bibles"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

SYSTEM_PROMPT_FILE = "Local_Prompt_Generator_System_Prompt.md"
BIBLE_FILES = [
    "Fruit_Character_Bible.md",
    "Fruit_Character_Acting_Guide.md",
    "Fruit_Character_Voice_Bible.md",
    "Fruit_Environment_Bible.md",
    "Character_Naming_Convention.md",
]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required prompt bible file: {path}") from exc


def load_context(bible_dir: Path) -> str:
    parts = [_read_text(bible_dir / SYSTEM_PROMPT_FILE)]
    for file_name in BIBLE_FILES:
        path = bible_dir / file_name
        parts.append(f"\n\n=== {file_name} ===\n{_read_text(path)}")
    return "\n".join(parts)


def _field(block: str, field_name: str) -> str:
    pattern = rf"\|\s*{re.escape(field_name)}\s*\|\s*([^|\n]+)\s*\|"
    match = re.search(pattern, block, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _heading_name(heading: str) -> str:
    name = re.sub(r"^\d+\.\s*", "", heading).strip()
    name = re.split(r"\s*\(|\s+-\s+|\s+—\s+", name, maxsplit=1)[0]
    return name.strip()


def _blocks_for_prompt(text: str, prompt_heading: str) -> list[dict[str, str]]:
    heading_pattern = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    entries: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]
        prompt_match = re.search(
            rf"\*\*{re.escape(prompt_heading)}:\*\*\s*\"(.+?)\"",
            block,
            re.DOTALL | re.IGNORECASE,
        )
        if not prompt_match:
            continue
        heading = match.group(1).strip()
        file_name = _field(block, "File name")
        display_name = _heading_name(heading)
        entries.append(
            {
                "heading": heading,
                "name": display_name,
                "file_name": file_name,
                "prompt": prompt_match.group(1).strip(),
            }
        )
    return entries


def _section_blocks(text: str) -> list[dict[str, str]]:
    heading_pattern = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    entries: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(1).strip()
        entries.append({"heading": heading, "name": _heading_name(heading), "block": text[start:end]})
    return entries


def _lookup(entries: list[dict[str, str]], name: str) -> dict[str, str] | None:
    wanted = _normalize(name)
    for entry in entries:
        aliases = {
            entry["name"],
            entry["file_name"],
            re.sub(r"Hero$", "", entry["file_name"]),
            re.sub(r"Hero$", "", entry["name"]),
        }
        if wanted in {_normalize(alias) for alias in aliases if alias}:
            return entry
    return None


def _lookup_section(context: str, name: str) -> dict[str, str] | None:
    wanted = _normalize(re.sub(r"Hero$", "", name))
    for entry in _section_blocks(context):
        aliases = {entry["name"], re.sub(r"Hero$", "", entry["name"])}
        if wanted in {_normalize(alias) for alias in aliases if alias}:
            return entry
    return None


def _character_entry(context: str, name: str) -> dict[str, str] | None:
    return _lookup(_blocks_for_prompt(context, "LOCKED IMAGE PROMPT"), name)


def _character_prompt(context: str, name: str) -> str | None:
    return (_character_entry(context, name) or {}).get("prompt")


def _environment_entry(context: str, name: str) -> dict[str, str] | None:
    return _lookup(_blocks_for_prompt(context, "LOCKED PROMPT"), name)


def _environment_prompt(context: str, name: str) -> str | None:
    return (_environment_entry(context, name) or {}).get("prompt")


def _value_after_label(block: str, label: str) -> str:
    match = re.search(rf"(?:- )?\*\*{re.escape(label)}:\*\*\s*(.+)", block, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _voice_profile(context: str, name: str) -> str:
    entry = _lookup(_blocks_for_prompt(context, "LOCKED VOICE PROFILE"), name)
    if not entry:
        raise SystemExit("NOT IN BIBLE - design once, then add to the Bible file before reuse.")
    return entry["prompt"]


def _acting_profile(context: str, name: str) -> dict[str, str]:
    entry = None
    wanted = _normalize(re.sub(r"Hero$", "", name))
    for section in _section_blocks(context):
        if wanted != _normalize(section["name"]):
            continue
        if _value_after_label(section["block"], "Movement style"):
            entry = section
            break
    if not entry:
        raise SystemExit("NOT IN BIBLE - design once, then add to the Bible file before reuse.")
    block = entry["block"]
    return {
        "movement": _value_after_label(block, "Movement style"),
        "gestures": _value_after_label(block, "Signature gestures"),
        "idle": _value_after_label(block, "Idle behavior"),
    }


def _choose_gesture(gestures: str, emotion: str) -> str:
    choices = [part.strip() for part in gestures.split("/") if part.strip()]
    if not choices:
        return "subtle matching gesture"
    emotion_l = emotion.lower()
    gesture_keywords = [
        ("hurt sad betrayed emotional sincere", ["heart", "chest", "eyes", "looks down"]),
        ("comfort forgive gentle loving", ["comfort", "shoulder", "open", "welcoming", "reaches"]),
        ("confident excited ambitious", ["points", "claps", "collar", "smirk"]),
        ("nervous unsure anxious caught", ["neck", "fidgets", "glances", "hunches"]),
        ("work determined resolve brave", ["fist", "brow", "sleeves", "chin up"]),
        ("cold judging dismissive", ["cuffs", "dismissive", "steepled", "collar"]),
    ]
    for emotion_words, wanted_words in gesture_keywords:
        if any(word in emotion_l for word in emotion_words.split()):
            for choice in choices:
                if any(word in choice.lower() for word in wanted_words):
                    return choice
    return choices[0]


def _character_or_die(context: str, name: str) -> dict[str, str]:
    entry = _character_entry(context, name)
    if not entry:
        raise SystemExit("NOT IN BIBLE - design once, then add to the Bible file before reuse.")
    return entry


def _environment_or_die(context: str, name: str) -> dict[str, str]:
    entry = _environment_entry(context, name)
    if not entry:
        raise SystemExit("NOT IN BIBLE - design once, then add to the Bible file before reuse.")
    return entry


def _split_names(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def call_ollama(context: str, user_request: str, *, model: str, ollama_url: str) -> str:
    full_prompt = (
        f"{context}\n\n"
        f"=== REQUEST ===\n{user_request}\n\n"
        f"=== YOUR OUTPUT (prompt text only) ===\n"
    )
    payload = json.dumps({"model": model, "prompt": full_prompt, "stream": False}).encode("utf-8")
    request = Request(ollama_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise SystemExit(f"Ollama request failed. Is Ollama running at {ollama_url}? {exc}") from exc
    return str(data.get("response", "")).strip()


def _deterministic_scene_prompt(context: str, args: argparse.Namespace) -> str:
    environment = _environment_or_die(context, args.environment)
    character_parts: list[str] = []
    for name in _split_names(args.characters):
        character = _character_or_die(context, name)
        character_parts.append(f"{character['file_name'] or character['name']}: {character['prompt']}")
    return (
        f"3D Pixar-style animated cartoon scene, {args.shot} shot, "
        f"BACKGROUND PLATE: {environment['prompt']}, "
        f"CHARACTERS: {'; '.join(character_parts)}, "
        f"MOMENT: {args.moment}, pose and expression match the emotional beat, locked character designs preserved, "
        f"storybook composition, vibrant saturated colors, high quality 3D animation render"
    )


def _listener_file_names(context: str, characters: str, speaker: str) -> str:
    listeners: list[str] = []
    speaker_norm = _normalize(re.sub(r"Hero$", "", speaker))
    for name in _split_names(characters):
        name_norm = _normalize(re.sub(r"Hero$", "", name))
        if name_norm == speaker_norm:
            continue
        character = _character_or_die(context, name)
        listeners.append(character["file_name"] or character["name"])
    return ", ".join(listeners) if listeners else "other characters"


def _deterministic_video_prompt(
    context: str,
    *,
    characters: str,
    environment_name: str,
    speaker: str,
    line: str,
    emotion: str,
    bgm: str,
    ambient: str,
    camera: str,
    mood: str,
) -> str:
    environment = _environment_or_die(context, environment_name)
    speaker_entry = _character_or_die(context, speaker)
    acting = _acting_profile(context, speaker)
    voice = _voice_profile(context, speaker)
    gesture = _choose_gesture(acting["gestures"], emotion)
    listeners = _listener_file_names(context, characters, speaker)
    return (
        f"3D Pixar-style animated cartoon clip, {camera}, "
        f"ENVIRONMENT: {environment['prompt']} + ambient motion, "
        f"SPEAKER: {speaker_entry['file_name'] or speaker_entry['name']} ({acting['movement']} + {gesture}) "
        f"- mouth moving in lip-sync - {voice} saying '{line}' - "
        f"{listeners} still and listening, BGM: {bgm}, AMBIENT: {ambient}, mood: {mood}, "
        f"high quality 3D animation render"
    )


def cmd_character(args: argparse.Namespace) -> None:
    context = load_context(args.bible_dir)
    result = _character_prompt(context, args.name)
    if not result:
        raise SystemExit("NOT IN BIBLE - design once, then add to the Bible file before reuse.")
    print(result)


def cmd_environment(args: argparse.Namespace) -> None:
    context = load_context(args.bible_dir)
    result = _environment_prompt(context, args.name)
    if not result:
        raise SystemExit("NOT IN BIBLE - design once, then add to the Bible file before reuse.")
    print(result)


def cmd_scene(args: argparse.Namespace) -> None:
    context = load_context(args.bible_dir)
    if not args.use_ollama:
        print(_deterministic_scene_prompt(context, args))
        return
    request = (
        "Generate a Phase 5 scene image prompt.\n"
        f"Characters present (file names): {args.characters}\n"
        f"Environment (file name): {args.environment}\n"
        f"Shot type: {args.shot}\n"
        f"Moment/emotional beat: {args.moment}\n"
        "Follow the Phase 5 image prompt formula, pulling locked character and environment details from the bibles."
    )
    print(call_ollama(context, request, model=args.model, ollama_url=args.ollama_url))


def cmd_video(args: argparse.Namespace) -> None:
    context = load_context(args.bible_dir)
    if not args.use_ollama:
        prompts = [
            _deterministic_video_prompt(
                context,
                characters=args.characters,
                environment_name=args.environment,
                speaker=args.speaker1,
                line=args.line1,
                emotion=args.emotion1,
                bgm=args.bgm,
                ambient=args.ambient,
                camera=args.camera,
                mood=args.mood,
            )
        ]
        if args.speaker2:
            prompts.append(
                _deterministic_video_prompt(
                    context,
                    characters=args.characters,
                    environment_name=args.environment,
                    speaker=args.speaker2,
                    line=args.line2 or "",
                    emotion=args.emotion2 or "",
                    bgm=args.bgm,
                    ambient=args.ambient,
                    camera=args.camera,
                    mood=args.mood,
                )
            )
        print("\n\n".join(prompts))
        return
    speaker2 = ""
    if args.speaker2:
        speaker2 = f"Speaker 2: {args.speaker2}, line: \"{args.line2 or ''}\", emotional beat: {args.emotion2 or ''}\n"
    request = (
        "Generate a Phase 6 video prompt.\n"
        f"Characters present (file names): {args.characters}\n"
        f"Environment (file name): {args.environment}\n"
        f"Speaker 1: {args.speaker1}, line: \"{args.line1}\", emotional beat: {args.emotion1}\n"
        f"{speaker2}"
        f"BGM mood cue: {args.bgm}\n"
        f"Ambient sound: {args.ambient}\n"
        f"Camera motion: {args.camera}\n"
        f"Overall mood: {args.mood}\n"
        "Apply the one-speaker-at-a-time rule and the exact Phase 6 merge formula."
    )
    print(call_ollama(context, request, model=args.model, ollama_url=args.ollama_url))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fruit Cartoon Channel local prompt generator")
    parser.add_argument("--bible-dir", type=Path, default=DEFAULT_BIBLE_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    sub = parser.add_subparsers(dest="command", required=True)

    character = sub.add_parser("character", help="Look up a locked character image prompt")
    character.add_argument("name", help="Character name or file name, e.g. Appy or AppyHero")
    character.set_defaults(func=cmd_character)

    environment = sub.add_parser("environment", help="Look up a locked environment prompt")
    environment.add_argument("name", help="Environment file name, e.g. VillageHomeExterior")
    environment.set_defaults(func=cmd_environment)

    scene = sub.add_parser("scene", help="Generate a Phase 5 scene image prompt")
    scene.add_argument("--characters", required=True)
    scene.add_argument("--environment", required=True)
    scene.add_argument("--shot", required=True)
    scene.add_argument("--moment", required=True)
    scene.add_argument("--use-ollama", action="store_true", help="Use Ollama instead of deterministic bible assembly")
    scene.set_defaults(func=cmd_scene)

    video = sub.add_parser("video", help="Generate a Phase 6 video prompt")
    video.add_argument("--characters", required=True)
    video.add_argument("--environment", required=True)
    video.add_argument("--speaker1", required=True)
    video.add_argument("--line1", required=True)
    video.add_argument("--emotion1", required=True)
    video.add_argument("--speaker2")
    video.add_argument("--line2")
    video.add_argument("--emotion2")
    video.add_argument("--bgm", required=True)
    video.add_argument("--mood", required=True)
    video.add_argument("--ambient", default="subtle village ambience")
    video.add_argument("--camera", default="gentle cinematic camera move")
    video.add_argument("--use-ollama", action="store_true", help="Use Ollama instead of deterministic bible assembly")
    video.set_defaults(func=cmd_video)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
