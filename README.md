# 2DVideo

Local-first pipeline for automated kids animation episodes.

The project plan lives in [`kids-channel-build-spec (1).md`](./kids-channel-build-spec%20(1).md) and is the source of truth for the build.

## Current status

This repository is being scaffolded from the build spec. The initial work focuses on:

- core schema files
- config registry files
- pipeline stage entry points
- orchestration layout

## Layout

```text
config/
pipeline/
schemas/
assets/
output/
logs/
```

## Notes

- The local repo is intentionally lightweight at first.
- Stage scripts are being created as explicit entry points before implementation details are filled in.
