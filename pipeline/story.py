from __future__ import annotations

import json
import logging
import urllib.request
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

log = logging.getLogger(__name__)


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
    available = _preferred_character_order(registry)
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


def _preferred_character_order(registry: dict) -> list[str]:
    imported = [cid for cid, entry in registry.items() if entry.get("source") or entry.get("render_mode")]
    cast = [cid for cid, entry in registry.items() if entry.get("tier") == "cast" and cid not in imported]
    guests = [cid for cid, entry in registry.items() if entry.get("tier") != "cast"]
    ordered = imported + cast + guests
    return ordered or list(registry.keys())


def _poem_lines() -> list[str]:
    return [
        "आओ मिलकर गाएं हम, खुशियों से भर जाएं हम!",
        "प्यारे पैर थिरकते जाएं, पेड़ की ठंडी छांव में।",
        "एक छोटा सा उछाल, एक प्यारी सी मुस्कान।",
        "घूम-घूम कर नाचें हम, मिलकर खेलें आज यहां।",
        "छोटे कदम और प्यारे दिल, सुर से सुर मिलाते जाएं।",
        "ताली बजाओ संग हमारे, आसमां के तारों तले।",
    ]


def _story_lines() -> list[str]:
    return [
        "आज हम सीखेंगे मिलकर खिलौने बांटना।",
        "एक दोस्त उदास होता है जब खिलौने छिन जाते हैं।",
        "अप्पी कहता है, 'हम बारी-बारी से खेलेंगे।'",
        "हर दोस्त एक-दूसरे की मदद करता है हमेशा।",
        "सभी मिलकर काम करें तो हर काम आसान होता है।",
        "चलो ताली बजाएं इस दोस्ती और एकता के लिए।",
        "चलो मिलकर गिनें तारे, एक, दो, तीन, चार।",
        "जब हम मिलकर काम करते हैं, खुशियां बढ़ती हैं अपार।",
        "बांटने से खुशियां बढ़ती हैं, दिल खिल उठते हैं हर बार।",
        "याद रखें: प्यार और धीरज ही है सच्ची दोस्ती का आधार।",
    ]


def _action_for_index(kind: str, index: int) -> str:
    if kind == "poem":
        return ["bounce_wave", "walk", "point", "dance", "jump", "clap", "laugh", "idle"][index % 8]
    return ["idle", "point", "walk", "dance", "clap", "bounce_wave", "walk", "point", "laugh", "sleep"][index % 10]


def _background_for_index(kind: str, index: int) -> str:
    if kind == "poem":
        return ["meadow_day", "forest_morning", "beach_day", "rainbow_yard", "park_sunny", "garden"][index % 6]
    return ["classroom", "library", "kitchen", "garden", "rainbow_yard", "forest_morning", "beach_day", "park_sunny", "meadow_day", "bedroom_night"][index % 10]


def _broll_prompt(index: int) -> str:
    prompts = [
        "magical flower field blooming in fast motion, soft pastel colors",
        "children's toy blocks stacking themselves into a rainbow tower",
        "sparkling stars drifting over a calm pastel sky",
        "butterflies dancing over a sunny meadow, gentle breeze",
        "soft rain falling on a cozy window, warm light inside",
    ]
    return prompts[index % len(prompts)]


# ---------------------------------------------------------------------------
# Ollama / LLM story generation
# ---------------------------------------------------------------------------

_OLLAMA_SYSTEM = (
    "You are a children's animation scriptwriter. "
    "You ONLY respond with valid JSON — no markdown, no extra text."
)


def _ollama_fill_episode(episode: dict, model: str, ollama_url: str) -> dict:
    """Ask an Ollama model to fill in dialogue lines and make the story coherent."""
    shots_summary = [
        {
            "shot_id": s["shot_id"],
            "characters": s.get("characters", []),
            "action": s.get("action", "idle"),
            "background": s.get("background", ""),
            "broll": s.get("broll", False),
        }
        for s in episode.get("shots", [])
    ]
    kind = episode.get("type", "poem")
    char_names = {
        cid: load_registry().get(cid, {}).get("name", cid)
        for cid in episode.get("characters_used", [])
    }
    prompt = (
        f"Create a fun, educational kids' {kind} for a Cocomelon-style cartoon. "
        f"Characters: {json.dumps(char_names)}. "
        f"Target duration: {episode.get('target_duration_sec', 150)} seconds. "
        f"Base template breakdown: {json.dumps(shots_summary)}. "
        "NOTE ON SHOT COUNT: You can choose to generate exactly as many shots as the story flow needs (e.g., 5 shots, 8 shots, etc.). You do NOT need to match the base template count. Do NOT add empty or placeholder B-roll shots at the end unless they are narratively useful. "
        "STRICT LANGUAGE REQUIREMENT: All character dialogue lines ('line') MUST be written in simple, beautiful Hindi (using Devanagari script) suitable for children. "
        "Return a JSON object with exactly these keys: "
        '"title" (string in English), '
        '"shots" (array of objects — one per shot, each with "shot_id" (string) and '
        '"dialogue" (array of {"character": id, "line": text in Hindi}) for non-broll shots, '
        'or "video_prompt" (string in English) for broll shots). '
        "Keep each dialogue line under 15 words. Rhyme where possible for poem type. "
        "Return ONLY the JSON."
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _OLLAMA_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
        }
    ).encode()
    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        content = body.get("message", {}).get("content", "")
        llm_data: dict = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        log.warning("Ollama call failed (%s). Using static episode.", exc)
        return episode

    # Merge LLM output into the base episode
    if isinstance(llm_data.get("title"), str):
        episode["title"] = llm_data["title"]

    llm_shots_list = llm_data.get("shots", [])
    if not isinstance(llm_shots_list, list) or not llm_shots_list:
        log.warning("Ollama returned empty or invalid shots list. Reverting to static.")
        return episode

    base_shots = episode.get("shots", [])
    base_by_id = {s.get("shot_id"): s for s in base_shots}
    
    new_shots = []
    registry = load_registry()
    
    for idx, llm_shot in enumerate(llm_shots_list):
        if not isinstance(llm_shot, dict):
            continue
        sid = llm_shot.get("shot_id", f"s{idx+1}")
        
        # Try to find corresponding base shot
        base_shot = base_by_id.get(sid)
        if not base_shot and idx < len(base_shots):
            base_shot = base_shots[idx]
            
        # Create a new shot merging base properties and LLM content
        shot = {}
        if base_shot:
            shot.update(base_shot)
        else:
            # Construct a fallback/extra shot dynamically
            shot = {
                "shot_id": sid,
                "characters": [episode["characters_used"][idx % len(episode["characters_used"])]],
                "action": _action_for_index(episode["type"], idx),
                "background": _background_for_index(episode["type"], idx),
                "duration_sec": 25,
                "broll": False
            }
            
        # Merge dialogue
        llm_dialogue = llm_shot.get("dialogue", [])
        if isinstance(llm_dialogue, list) and llm_dialogue:
            merged: list[dict] = []
            for item in llm_dialogue:
                if not isinstance(item, dict):
                    continue
                cid = item.get("character", "")
                if cid not in registry:
                    cid = (shot.get("characters") or episode["characters_used"])[0]
                line = str(item.get("line", "")).strip()
                if line:
                    merged.append({
                        "character": cid,
                        "line": line,
                        "voice_id": registry[cid]["voice_id"]
                    })
            if merged:
                shot["dialogue"] = merged
                shot["broll"] = False
                shot.pop("video_prompt", None)
            else:
                # No valid dialogue lines -> make B-roll
                shot["dialogue"] = []
                shot["broll"] = True
                shot["video_prompt"] = llm_shot.get("video_prompt") or "bright pastel children's cartoon background"
                shot.pop("action", None)
                shot["characters"] = []
        else:
            # No dialogue -> make B-roll
            shot["dialogue"] = []
            shot["broll"] = True
            shot["video_prompt"] = llm_shot.get("video_prompt") or "bright pastel children's cartoon background"
            shot.pop("action", None)
            shot["characters"] = []
            
        # Enforce shot ID naming structure
        shot["shot_id"] = f"s{idx+1}"
        new_shots.append(shot)
        
    episode["shots"] = new_shots
    return episode


def build_episode_with_ollama(
    template: dict,
    *,
    episode_id: str | None = None,
    title: str | None = None,
    kind: str | None = None,
    characters: list[str] | None = None,
    model: str = "gemma3:12b",
    ollama_url: str = "http://localhost:11434",
) -> dict:
    """Build an episode using the LLM for richer dialogue; falls back to static content."""
    base = build_episode(
        template=template,
        episode_id=episode_id,
        title=title,
        kind=kind,
        characters=characters,
    )
    filled = _ollama_fill_episode(base, model, ollama_url)
    # Re-validate after LLM merge
    try:
        validate_json_schema("episode", filled)
    except Exception:  # noqa: BLE001
        log.warning("LLM output failed schema validation; reverting to static episode.")
        return base
    return filled
