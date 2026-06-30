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
- review finished episodes

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
