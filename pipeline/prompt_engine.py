"""
prompt_engine.py — Gemma 3 12B Scene Prompt Generator
-------------------------------------------------------
Calls Ollama (gemma3:12b) to produce stable-diffusion-compatible
image_prompt + negative_prompt for each shot, using locked character
descriptor strings from the character catalog so the output stays
visually consistent across all shots.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character visual descriptors (locked — always injected into every prompt)
# These ensure SD keeps the same character look across shots.
# ---------------------------------------------------------------------------

CHARACTER_DESCRIPTORS: dict[str, str] = {
    "char_appy":   "A cute 2D flat illustration of a cartoon apple character named Appy, round glossy red-and-green apple body, small green leaf sprouting from the top of the head like hair, warm brown eyes, friendly round face, rosy cheeks, joyful sincere smile",
    "char_nana":   "A cute 2D flat illustration of a cartoon banana character named Nana, long curved yellow banana body, friendly round face, big expressive warm brown eyes, a joyful smile, white soft underbelly",
    "char_mango":  "A cute 2D flat illustration of a cartoon mango character named Mango, energetic plump golden-yellow mango body, small stem on top, warm sparkle eyes, playful grin",
    "char_berry":  "A cute 2D flat illustration of a cartoon grape character named Berry, shy round purple grape body, small green leaf hat, gentle expressive eyes, soft lavender skin",
    "char_orangy": "A cute 2D flat illustration of a cartoon orange character named Orangy, bubbly round orange body, textured dimpled skin, wide cheerful smile, bright orange color, big friendly eyes",
    "char_pinny":  "A cute 2D flat illustration of a cartoon pineapple character named Pinny, tall spiky pineapple body, diamond-pattern yellow skin, green crown leaves on head, sweet shy smile",
    "char_pappy":  "A cute 2D flat illustration of a cartoon papaya character named Pappy, wise elder papaya body, salmon-orange skin, kind fatherly eyes, gentle expression, plump oval shape",
    "char_kiwi":   "A cute 2D flat illustration of a cartoon kiwi character named Kiwi, small fuzzy brown kiwi body, bright green cross-section pattern on belly, surprised expressive eyes",
    "char_lime":   "A cute 2D flat illustration of a cartoon lime character named Lime, zesty bright green lime body, sharp happy eyes, tiny star-like stem crown, energetic pose",
    "char_cherry": "A cute 2D flat illustration of cartoon twin cherry characters named Cherry joined by a green stem, bright red round bodies, giggling expressive faces",
    "char_peary":  "A cute 2D flat illustration of a cartoon pear character named Peary, gentle green pear body, soft round bottom, small stem hat, kind motherly eyes, pastel green skin",
}

# Shared negative prompt — highly detailed to avoid model off-character deviations
BASE_NEGATIVE = (
    "ugly, deformed, blurry, extra limbs, extra fingers, mutated hands, bad anatomy, "
    "asymmetrical face, watermark, text, logo, signature, brand names, disfigured, horror, "
    "scary, realistic, photorealistic, 3d render, nsfw, violence, dark, grim, gray background, "
    "white background, off-model, inconsistent character design, muted colors, low detail"
)

# Style suffix appended to every positive prompt
STYLE_SUFFIX = (
    "preschool children's animation style, vibrant pastel colors, soft lighting, "
    "clean bold outlines, 2D flat illustration, cheerful bright atmosphere, kids show"
)


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def _ollama_generate(
    prompt: str,
    *,
    model: str = "gemma3:12b",
    url: str = "http://localhost:11434",
    timeout: int = 60,
) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 300},
    }).encode()
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception as exc:
        log.warning("[prompt_engine] Ollama call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_scene_prompt(
    *,
    character_ids: list[str],
    action: str,
    background: str,
    dialogue: str,
    format: str = "full",          # "short" (9:16) or "full" (16:9)
    use_ollama: bool = True,
    ollama_model: str = "gemma3:12b",
    ollama_url: str = "http://localhost:11434",
) -> dict[str, str]:
    """
    Returns {"image_prompt": ..., "negative_prompt": ...} for a single shot.

    If Ollama is unavailable or fails, falls back to a deterministic template.
    """
    char_descs = [
        CHARACTER_DESCRIPTORS.get(cid, f"a cartoon fruit character named {cid}")
        for cid in character_ids
        if cid
    ]
    char_text = ", ".join(char_descs) if char_descs else "a cute cartoon fruit character"

    orientation = "portrait orientation, vertical composition" if format == "short" else "landscape orientation, wide composition"

    if use_ollama:
        llm_prompt = (
            f"You are an expert Stable Diffusion prompt writer for a premium preschool children's cartoon show.\n"
            f"Write a highly detailed, descriptive, SD-optimized scene prompt based on this input:\n\n"
            f"Character Details: {char_text}\n"
            f"Character Action: {action}\n"
            f"Background Setting: {background}\n"
            f"Scene Dialogue Context: {dialogue}\n"
            f"Orientation Layout: {orientation}\n\n"
            f"Rules:\n"
            f"- Output ONLY the prompt itself. Do not write any introduction, notes, or extra labels.\n"
            f"- Start directly with: 'A cute 2D flat illustration of...'\n"
            f"- Describe the character's body, friendly face expression, pose, and active gesture based on the Character Action.\n"
            f"- Describe the background setting detailed and child-friendly.\n"
            f"- Ensure style guidelines are incorporated: 'preschool children's animation style, vibrant pastel colors, soft lighting, clean bold outlines, cheerful bright atmosphere'.\n"
            f"- Keep the length between 60 to 90 words.\n"
        )
        raw = _ollama_generate(llm_prompt, model=ollama_model, url=ollama_url)
        if raw:
            image_prompt = f"{raw}, {orientation}, high detail"
            return {"image_prompt": image_prompt, "negative_prompt": BASE_NEGATIVE}

    # Deterministic fallback matching the improved premium template
    image_prompt = (
        f"{char_text} doing {action}, in a {background}, {STYLE_SUFFIX}, {orientation}, high detail"
    )
    return {"image_prompt": image_prompt, "negative_prompt": BASE_NEGATIVE}


def enrich_episode_with_prompts(
    episode: dict,
    *,
    format: str = "full",
    use_ollama: bool = True,
    ollama_model: str = "gemma3:12b",
    ollama_url: str = "http://localhost:11434",
) -> dict:
    """
    Iterates over all shots in episode JSON and injects image_prompt +
    negative_prompt into each shot that is NOT a pure broll shot.
    Returns the mutated episode dict.
    """
    for shot in episode.get("shots", []):
        if shot.get("broll") and not shot.get("characters"):
            # Pure broll — keep existing video_prompt, skip SD prompt
            continue
        chars = shot.get("characters", [])
        action = shot.get("action", "standing happily")
        background = shot.get("background", "colorful garden")
        dialogue_lines = shot.get("dialogue", [])
        dialogue_hint = dialogue_lines[0].get("line", "") if dialogue_lines else ""

        # Keep user custom prompts if already set
        if not shot.get("image_prompt") or not shot.get("negative_prompt"):
            prompts = build_scene_prompt(
                character_ids=chars,
                action=action,
                background=background,
                dialogue=dialogue_hint,
                format=format,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                ollama_url=ollama_url,
            )
            if not shot.get("image_prompt"):
                shot["image_prompt"] = prompts["image_prompt"]
            if not shot.get("negative_prompt"):
                shot["negative_prompt"] = prompts["negative_prompt"]
        log.info("[prompt_engine] shot=%s prompt=%s", shot.get("shot_id"), shot.get("image_prompt", "")[:60])

    episode["format"] = format
    return episode
