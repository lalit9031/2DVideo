from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pipeline.bootstrap import bootstrap_demo_cast
from pipeline.common import (
    load_registry,
    read_json,
    slugify,
    today_iso,
    validate_json_schema,
    write_json,
)


@dataclass(frozen=True)
class StoryTemplate:
    type: str
    target_duration_sec: int
    notes: list[str]
    structure: str


def load_template(path: Path) -> dict:
    data = read_json(path)
    if not isinstance(data, dict):
        raise TypeError("Episode template must be a JSON object.")
    return data


def build_episode(
    *,
    template: dict,
    episode_id: str | None = None,
    title: str | None = None,
    kind: str | None = None,
    characters: list[str] | None = None,
) -> dict:
    registry = load_registry()
    if not registry:
        bootstrap_demo_cast()
        registry = load_registry()
    available = list(registry.keys())
    kind = kind or template.get("type", "poem")
    target_duration = int(template.get("target_duration_sec", 150 if kind == "poem" else 300))
    episode_id = episode_id or f"{today_iso()}-{kind}-01"
    title = title or template.get("title") or ("The Friendly Adventure" if kind == "story" else "The Happy Song")
    if not characters:
        if kind == "poem":
            characters = available[:2] if len(available) >= 2 else available[:1]
        else:
            characters = available[:3] if len(available) >= 3 else available
    if not characters:
        raise ValueError("No characters available to build an episode.")
    shots: list[dict] = []
    shot_count = 6 if kind == "poem" else 10
    base_duration = max(4, target_duration // shot_count)
    line_bank = _poem_lines() if kind == "poem" else _story_lines()
    for index in range(shot_count):
        characters_in_shot = [characters[index % len(characters)]]
        if kind == "story" and len(characters) > 1 and index % 3 == 1:
            characters_in_shot.append(characters[(index + 1) % len(characters)])
        line = line_bank[index % len(line_bank)]
        active = characters_in_shot[0]
        shot = {
            "shot_id": f"s{index + 1}",
            "characters": characters_in_shot,
            "action": _action_for_index(kind, index),
            "dialogue": [
                {
                    "character": active,
                    "line": line,
                    "voice_id": registry[active]["voice_id"],
                }
            ],
            "background": _background_for_index(kind, index),
            "duration_sec": base_duration if index < shot_count - 1 else max(3, target_duration - base_duration * (shot_count - 1)),
            "broll": False,
        }
        if kind == "story" and index in {3, 7}:
            shot["broll"] = True
            shot["video_prompt"] = _broll_prompt(index)
            shot["characters"] = []
            shot["dialogue"] = []
            shot.pop("action", None)
        shots.append(shot)
    if kind == "poem" and len(shots) >= 2:
        shots[-1]["broll"] = True
        shots[-1]["video_prompt"] = "bright pastel celebration flowers blooming in a cheerful meadow"
        shots[-1]["characters"] = []
        shots[-1]["dialogue"] = []
        shots[-1].pop("action", None)
    episode = {
        "episode_id": episode_id,
        "type": kind,
        "target_duration_sec": target_duration,
        "title": title,
        "characters_used": characters,
        "shots": shots,
        "generated_from_template": template.get("structure", ""),
        "created": today_iso(),
    }
    validate_json_schema("episode", episode)
    return episode


def _poem_lines() -> list[str]:
    return [
        "Hop along and sing with me, bright as bright can be!",
        "Little feet go tap, tap, tap, under the sunny tree.",
        "One small bounce, one happy cheer, everybody smiles today.",
        "Round and round we wiggle now, in a gentle play.",
        "Tiny paws and happy hearts make a joyful tune.",
        "Clap along and spin with us beneath the moon of noon.",
    ]


def _story_lines() -> list[str]:
    return [
        "Today we learn to share the toys so everyone can play.",
        "One friend feels a little sad when the blocks get taken away.",
        "Bibi says, 'We can take turns and make a fair little plan.'",
        "Then each friend helps the other out, just like a good friend can.",
        "The puzzle gets completed when everyone lends a hand.",
        "We clap for kind teamwork now; that is how we understand.",
        "Next we count the stars together, one, two, three, and four.",
        "When we work as a happy team, we always find much more.",
        "Sharing makes the fun grow big and helps our hearts feel bright.",
        "Let's remember: kind and patient friends make every day feel right.",
    ]


def _action_for_index(kind: str, index: int) -> str:
    if kind == "poem":
        return ["bounce_wave", "walk", "point", "idle", "jump", "sleep"][index % 6]
    return ["idle", "point", "walk", "bounce_wave", "idle", "jump", "walk", "point", "idle", "sleep"][index % 10]


def _background_for_index(kind: str, index: int) -> str:
    if kind == "poem":
        return ["meadow_day", "tree_path", "sunny_hill", "cloud_field"][index % 4]
    return ["classroom", "playroom", "garden", "rainbow_yard", "storybook_room"][index % 5]


def _broll_prompt(index: int) -> str:
    prompts = [
        "magical flower field blooming in fast motion, soft pastel colors",
        "children's toy blocks stacking themselves into a rainbow tower",
        "sparkling stars drifting over a calm pastel sky",
    ]
    return prompts[index % len(prompts)]

