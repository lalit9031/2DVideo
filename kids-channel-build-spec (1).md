# Build Spec: Automated Kids Animation Channel (Cocomelon-style)

**Target hardware:** AMD Ryzen 9 7900X, RX 7900 XTX (24GB VRAM, ROCm), 64GB RAM, Ubuntu 24.04
**Output formats:** Poems/rhymes (2-3 min), Learning stories (5-10 min)
**Engine:** 2D rigged puppet animation (primary) + sparing local AI video-gen for B-roll
**This document is written to be handed to a coding agent (e.g. Claude Code) to scaffold and build.**

---

## 1. Tech stack (concrete choices)

| Stage | Tool | Why |
|---|---|---|
| Story/script generation | Gemma 3 12B via Ollama (local) | Fits 24GB VRAM, fast, structured JSON output |
| Character art | Existing `sd-server-rocm` pipeline | Already built; generate part-separated character sheets |
| Character rig + animation | Custom Python rig system (Pillow/OpenCV compositing + bone hierarchy + keyframes) | Fully scriptable, no GUI dependency, bot can generate/modify it directly — far easier to automate than Spine/Live2D |
| Voice synthesis (multi-character cloning) | XTTS-v2 (Coqui) | Mature, supports speaker cloning from short reference clips, runs on ROCm |
| Lip-sync | Phoneme/word timestamps from TTS (or `whisper-timestamped` as fallback) mapped to a small viseme set (5-8 mouth shapes) | No separate lip-sync model needed — mouth shapes are part of the rig, swapped on timing |
| AI video B-roll (sparing use) | LTX-2.3 (local, ROCm) | Already tested locally; used only for a small number of "wow" shots per episode |
| Upscaling | Real-ESRGAN (local) | Final pass on rendered output |
| Final assembly | ffmpeg (scripted, filter graph) | Combine animation render + dialogue + music + SFX |
| Orchestration | Python orchestrator + systemd timer (or cron) | Daily unattended run, retries, logging |
| Publishing (optional, later) | YouTube Data API v3 | Only after the pipeline is stable |
| Guest character art (auto-generated) | `sd-server-rocm` + a "house style" LoRA (separate from your 5 cast LoRAs) | Lets the pipeline generate new one-off characters that still visually match your show |
| Character/voice registry | `config/characters/registry.json` (flat lookup file) | Single source of truth the automation reads to know which voice belongs to which character |

---

## 1b. Character tiers — your 5 cast vs. auto-generated guests

You'll provide 5 main characters that recur across every episode. The system needs to treat these very differently from any one-off characters it generates on its own.

**Tier 1 — Your Cast (the 5 characters you provide)**
- Manually locked: art, voice, and rig are set up once and never auto-regenerated
- Each gets its own identity LoRA (trained on the locked design) so the system can generate *new poses/expressions* for that character later without it drifting off-model
- These are the only characters guaranteed to appear in every episode

**Tier 2 — Guest/one-off characters (system-generated, optional)**
- Used for minor side characters in individual stories
- Generated using the shared `house_style` LoRA (trained on your cast's overall art style — palette, line weight, proportions) so guests still look like they belong in your show, without needing a dedicated LoRA per guest
- Get a *minimal* rig (usually just idle + mouth-flap, not the full cast animation set)
- Logged into `registry.json` on first creation — if the story engine reuses that guest character later, it's pulled from the registry instead of regenerated, so even guests stay consistent across the episodes they appear in

You can ship the whole channel using Tier 1 only (your 5 characters in every story) and add Tier 2 later — it's an optional extension once the core pipeline is stable.

---

## 2. Repo structure

```
kids-channel/
├── config/
│   ├── characters/              # one JSON + asset folder per character (cast AND guests)
│   │   ├── registry.json        # MASTER LOOKUP — every character_id -> voice_id -> asset paths
│   │   ├── char_01_bunny.json
│   │   └── char_01_bunny/       # part PNGs: head.png, eyes_open.png, eyes_blink.png,
│   │                            # mouth_closed.png, mouth_open.png, mouth_wide.png,
│   │                            # arm_l.png, arm_r.png, leg_l.png, leg_r.png, body.png
│   ├── voices/                  # cloned voice reference clips, one per character
│   │   ├── char_01_bunny_ref.wav
│   │   └── voice_registry.json  # voice_id -> reference clip path + TTS settings
│   ├── loras/
│   │   ├── house_style.safetensors    # overall show art style, used for guest characters
│   │   └── char_01_bunny.safetensors  # per-cast-character identity LoRA (Tier 1 only)
│   └── episode_templates/       # poem template, learning-story template (prompt scaffolds)
├── pipeline/
│   ├── 01_story_engine.py       # Gemma 3 12B → episode JSON (script + shot list)
│   ├── 02_asset_check.py        # checks registry.json for every character in the episode;
│   │                            #   known → pass through, unknown → triggers 02b
│   ├── 02b_guest_character_gen.py  # auto-generates art + rig + voice clone for a NEW
│   │                            #   character, then writes it into registry.json
│   ├── 03_tts.py                 # reads each dialogue line's character_id, looks up voice_id
│   │                            #   in voice_registry.json, synthesizes with XTTS-v2
│   ├── 04_rig_render.py          # renders 2D puppet animation per shot from rig + audio timing
│   ├── 05_broll_gen.py           # LTX-2.3 local gen, only for shots flagged "broll": true
│   ├── 06_upscale.py             # Real-ESRGAN pass
│   ├── 07_assemble.py            # ffmpeg: stitch shots + audio + music/SFX → final MP4
│   └── orchestrator.py           # runs 01-07 in order, retries, logs, writes daily summary
├── assets/
│   ├── music/                    # royalty-free background tracks, looped
│   └── sfx/                      # short sound effects library
├── output/
│   ├── YYYY-MM-DD/               # per-day working files
│   └── final/                    # final MP4s ready for upload
└── logs/
```

---

## 3. Data schemas (lock these first — everything downstream depends on them)

### `episode.json` (output of story engine, input to everything else)
```json
{
  "episode_id": "2026-07-01-poem-01",
  "type": "poem",                 // "poem" | "story"
  "target_duration_sec": 150,
  "title": "The Bouncy Bunny Song",
  "characters_used": ["char_01_bunny", "char_02_fox"],
  "shots": [
    {
      "shot_id": "s1",
      "characters": ["char_01_bunny"],
      "action": "bounce_wave",      // maps to a predefined rig animation cycle
      "dialogue": [
        {"character": "char_01_bunny", "line": "Hop, hop, hop, little bunny!", "voice_id": "char_01_bunny"}
      ],
      "background": "meadow_day",
      "duration_sec": 6,
      "broll": false
    },
    {
      "shot_id": "s7",
      "broll": true,
      "video_prompt": "magical flower field blooming in fast motion, soft pastel colors",
      "duration_sec": 5
    }
  ]
}
```

### `character.json` (one per character — locks consistency)
```json
{
  "character_id": "char_01_bunny",
  "name": "Bibi the Bunny",
  "voice_id": "char_01_bunny",
  "rig_parts": ["head", "eyes_open", "eyes_blink", "mouth_closed", "mouth_open", "mouth_wide",
                "body", "arm_l", "arm_r", "leg_l", "leg_r"],
  "anim_cycles": ["idle", "bounce_wave", "walk", "point", "jump", "sleep"],
  "pivot_points": { "head": [0, -120], "arm_l": [-60, -20], "arm_r": [60, -20] }
}
```

Lock these two schemas in Phase 1 before building anything else — every later stage reads/writes them.

---

## 3b. How the automation knows which voice belongs to which character (this is the key linking mechanism)

This is the chain that makes multi-voice automation work, end to end:

1. **You provide one reference voice clip per cast character** (10-30 seconds of clean speech is enough for XTTS-v2 cloning) — saved as `config/voices/char_01_bunny_ref.wav`, etc.

2. **`voice_registry.json` maps `voice_id` → reference clip + settings:**
```json
{
  "char_01_bunny": {
    "reference_clip": "config/voices/char_01_bunny_ref.wav",
    "language": "en",
    "pitch_shift": 0,
    "speed": 1.0
  },
  "char_02_fox": {
    "reference_clip": "config/voices/char_02_fox_ref.wav",
    "language": "en",
    "pitch_shift": -2,
    "speed": 0.95
  }
}
```

3. **`character.json` for each character stores its `voice_id`** (shown in §3 above) — this is what links a *character* to a *voice*. In the simple case `voice_id` == `character_id`, but keeping them as separate fields means two characters could ever share a voice, or you could swap a character's voice later without renaming the character.

4. **The story engine (`01_story_engine.py`) writes `character` (not raw text) into every dialogue line** in `episode.json` — it never invents free-floating audio, every line is explicitly attributed to a known `character_id`.

5. **`03_tts.py` does the actual lookup, per line, automatically:**
   - Read `dialogue[i].character` from the shot
   - Look up that `character_id` in `registry.json` → get its `voice_id`
   - Look up that `voice_id` in `voice_registry.json` → get the reference clip + settings
   - Call XTTS-v2 with `(line text, reference_clip, language, pitch_shift, speed)` → outputs `shot_id_line_i.wav` + word/phoneme timestamps
   - If a `character_id` has no matching entry in either registry, the stage fails loudly and flags the episode rather than silently generating a default/wrong voice — you want errors here to be visible, not silently wrong audio

6. **Those per-line audio files + timestamps feed two places:** the final audio mix (`07_assemble.py`) and the mouth-shape timing for that character's rig render (`04_rig_render.py`) — so lip-sync, voice identity, and character identity are all driven by the same `character_id` key throughout the whole pipeline. One ID, traced through every stage — that's what keeps 5 characters' worth of audio from ever getting crossed.

---

### `registry.json` (master lookup — every character that exists, cast or guest)
```json
{
  "char_01_bunny": {
    "tier": "cast",
    "name": "Bibi the Bunny",
    "voice_id": "char_01_bunny",
    "character_json": "config/characters/char_01_bunny.json",
    "created": "2026-07-01"
  },
  "char_07_owl_guest": {
    "tier": "guest",
    "name": "Ollie the Owl",
    "voice_id": "char_07_owl_guest",
    "character_json": "config/characters/char_07_owl_guest.json",
    "created": "2026-07-15",
    "source_episode": "2026-07-15-story-02"
  }
}
```
`02_asset_check.py` reads this file first for every character referenced in an episode — known `character_id` → proceed; unknown → trigger `02b_guest_character_gen.py`, which generates art/rig/voice and writes a new entry here before continuing.

---

## 4. Build phases (checklist format — each phase has a clear "done" condition)

### Phase 0 — Environment setup
- [ ] Install Ollama, pull `gemma3:12b`
- [ ] Install XTTS-v2 + ROCm-compatible PyTorch, verify GPU inference works
- [ ] Confirm LTX-2.3 local install still working (already done per your testing)
- [ ] Install Real-ESRGAN, ffmpeg, Pillow/OpenCV
- **Done when:** each tool runs a basic "hello world" generation independently

### Phase 1 — Lock schemas + build your 5 cast characters
- [ ] Finalize `episode.json`, `character.json`, `registry.json`, `voice_registry.json` schemas (above, or bot-adjusted version)
- [ ] For each of your 5 characters: generate part-separated art via existing SD pipeline (front-facing, neutral pose, transparent PNGs: head, 3 mouth shapes, 2 eye states, body, 2 arms, 2 legs)
- [ ] For each character: record/provide a clean 10-30s reference voice clip, add an entry to `voice_registry.json`, test XTTS-v2 cloning sounds distinct from the other 4
- [ ] Write each character's `character.json` (with matching `voice_id`) and register all 5 in `registry.json`
- **Done when:** all 5 cast characters have a complete `character.json` + asset folder + a working, distinguishable cloned voice, all linked through `registry.json`/`voice_registry.json`

### Phase 1c (optional) — Guest character auto-generation
- [ ] Train the `house_style` LoRA from your 5 cast characters' art
- [ ] Build `02b_guest_character_gen.py`: takes a short appearance description from the story engine → generates turnaround art with house-style LoRA → extracts minimal rig → clones a one-off voice (or assigns from a small pre-recorded pool of generic voices) → writes new entry to `registry.json`
- **Done when:** the story engine can reference a brand-new character name it invented, and the pipeline auto-creates working art/rig/voice for it without manual intervention

### Phase 2 — Rig + render engine
- [ ] Build the Python rig system: bone hierarchy, keyframe interpolation (position/rotation/scale per part over time), frame compositor (Pillow/OpenCV)
- [ ] Implement 3-4 base animation cycles (idle, bounce_wave, walk, point) as reusable keyframe templates
- [ ] Implement mouth-shape swapping driven by phoneme/word timestamps from TTS output
- **Done when:** you can feed one character + one line of dialogue + audio → get a rendered MP4 clip with synced mouth movement

### Phase 3 — Story engine
- [ ] Write the poem-template prompt and learning-story-template prompt for Gemma 3 12B (structured JSON output matching `episode.json`)
- [ ] Validate output against schema; reject/retry malformed output
- **Done when:** running the story engine produces a valid `episode.json` for both a poem and a story, using only characters that already exist

### Phase 4 — Full shot pipeline (no AI video yet)
- [ ] Wire 01→02→03→04→06→07 (skip broll initially) for a single full episode
- [ ] Add background art (static or slow pan/zoom) per shot
- [ ] Add music/SFX layering in the assembly step
- **Done when:** a complete 2-3 min poem renders end-to-end, unattended, from a single command

### Phase 5 — Sparing AI video B-roll
- [ ] Add LTX-2.3 generation only for shots flagged `"broll": true`
- [ ] Cap B-roll shots per episode (see compute budget, §5) so daily runs stay within your local GPU's realistic throughput
- **Done when:** an episode with 1-2 B-roll shots renders end-to-end within your time budget

### Phase 6 — Multi-character episodes + learning stories
- [ ] Test a multi-character 5-10 min learning story end-to-end using all 5 cast characters (already built in Phase 1)
- [ ] Verify each character's audio is correctly attributed — spot-check that no two characters' lines ever get the wrong `voice_id`
- **Done when:** all 5 characters work together in one episode with distinct, correctly-matched voices

### Phase 7 — Orchestration
- [ ] Build `orchestrator.py`: runs N episodes/day (configurable), retries failed stages, logs per-stage timing, writes a daily summary
- [ ] Schedule via systemd timer or cron
- [ ] Add a fail-safe: if a stage fails twice, skip that episode and alert rather than blocking the whole run
- **Done when:** the pipeline runs unattended for 7 consecutive days producing valid output

### Phase 8 (optional, later) — Publishing
- [ ] YouTube Data API upload script, only after Phase 7 is stable for at least a week

---

## 5. Compute budget (why B-roll must stay capped)

Based on your tested LTX-2.3 speed (40 min compute per 10s output):

| Episode type | Target length | Suggested AI B-roll | B-roll compute cost | Rest (rigged animation) |
|---|---|---|---|---|
| Poem | 2-3 min | 1 shot, 5s | ~20 min | Seconds (compositing only) |
| Learning story | 5-10 min | 2-3 shots, 5s each | ~40-60 min | Seconds (compositing only) |

At 3-4 episodes/day mixing poems and stories, total daily B-roll compute lands roughly **1.5-3 hours/day** — comfortably inside a day on your GPU, instead of the 48-64 hours/day the original "every frame is AI video" plan required. This is the number that makes daily automation actually possible.

---

## 6. Content templates (feed these into the story engine prompts)

**Poem/rhyme structure (2-3 min):** simple AABB or ABAB rhyme scheme, repetitive chorus (kids' content relies on repetition), one clear simple action per verse mapped to an existing anim cycle, upbeat tempo.

**Learning story structure (5-10 min):** clear single learning goal (counting, colors, sharing, etc.), 3-5 act structure (setup → small problem → character tries solutions → resolution → recap of the lesson), repeated key phrase/lesson line for retention, gentle pacing.

Encode both as explicit prompt templates in `config/episode_templates/` so the story engine output is consistent and schema-valid every time.

---

## 7. Handing this to a coding agent

This spec is structured so each phase is independently buildable and testable. Recommended order: hand Phase 0-2 to a coding agent first — environment setup, your 5 cast characters (art + voice + registry), and the rig/render engine — before touching Phase 1c (guest auto-generation) or Phase 5 (AI video B-roll). Getting the core rig/render + voice-mapping right once, on your real cast, is more valuable than building the guest-character extension early.
