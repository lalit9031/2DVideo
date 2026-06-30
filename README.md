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
output/
logs/
```

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

## Notes

- The local repo is intentionally lightweight at first.
- Stage scripts are being created as explicit entry points before implementation details are filled in.
