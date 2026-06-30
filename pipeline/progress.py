from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROGRESS_ENV = "2DVIDEO_PROGRESS_FILE"


def progress_path_from_env() -> Path | None:
    value = os.environ.get(PROGRESS_ENV)
    if not value:
        return None
    return Path(value)


def read_progress(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_progress(path: Path | None, *, percent: float, stage: str = "", message: str = "", status: str = "running") -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "percent": max(0.0, min(100.0, float(percent))),
        "stage": stage,
        "message": message,
        "status": status,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

