# 2DVideo

Local-first pipeline for automated kids animation episodes.

The project plan lives in [`kids-channel-build-spec (1).md`](./kids-channel-build-spec%20(1).md) and is the source of truth for the build.

## Current status

This repository is being scaffolded from the build spec. The initial work focuses on:

- core schema files
- config registry files
- pipeline stage entry points
- orchestration layout
- local browser UI for testing pipeline runs

## Layout

```text
config/
pipeline/
ui/
schemas/
assets/
prompt_bibles/
tools/
output/
logs/
```

## Prompt Bibles

The locked fruit-channel reference material lives in `prompt_bibles/`:

- `Fruit_Character_Bible.md`
- `Fruit_Character_Acting_Guide.md`
- `Fruit_Character_Voice_Bible.md`
- `Fruit_Environment_Bible.md`
- `Character_Naming_Convention.md`
- `Local_Prompt_Generator_System_Prompt.md`

Use the local prompt generator for Phase 3-6 prompts:

```bash
python3 tools/prompt_generator.py character Appy
python3 tools/prompt_generator.py environment VillageHomeExterior
python3 tools/prompt_generator.py scene --characters Appy,Ozzy --environment VillageHomeExterior --shot "close-up" --moment "Appy realizes Ozzy lied"
python3 tools/prompt_generator.py video --characters Appy,Ozzy --environment VillageHomeExterior --speaker1 Appy --line1 "Tumne mujhse jhoot kyun bola?" --emotion1 "hurt, quietly betrayed" --bgm "Betrayal reveal" --mood heartbreaking
```

By default, every command assembles from the locked bible text directly. Add `--use-ollama` to `scene` or `video` when you want local model judgment; override the model/API with `OLLAMA_MODEL` or `OLLAMA_URL` when needed.

## UI

Start the local test UI with:

```bash
python3 -m ui.app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` to:

- run the orchestrator
- run individual stages
- import a character sheet
- inspect registries and generated outputs
- review finished episodes
- generate locked character, environment, scene, and video prompts from `/prompts`

### Review finished output

After a run completes, open the review page from the home screen or go directly to:

```text
http://127.0.0.1:8000/review
```

The review page shows:

- the final MP4
- assembled audio
- episode and stage JSON files
- the shot list and manifest summary

## Notes

- The local repo is intentionally lightweight at first.
- Stage scripts are being created as explicit entry points before implementation details are filled in.
