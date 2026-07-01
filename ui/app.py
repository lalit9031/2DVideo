from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import mimetypes
import wave
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlparse

from pipeline.backgrounds import generate_background
from pipeline.common import write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
UI_PROGRESS_FILE = OUTPUT_DIR / ".ui_progress.json"
CONFIG_DIR = ROOT / "config"
CHAR_DIR = ROOT / "config" / "characters"
VOICE_DIR = ROOT / "config" / "voices"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MEDIA_SUFFIXES = {".mp4", ".wav", ".json", ".txt", ".log"}


@dataclass
class JobState:
    status: str = "idle"
    title: str = "No job running"
    command: list[str] = field(default_factory=list)
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    progress: float | None = None
    stage: str = ""
    message: str = ""
    output: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "title": self.title,
            "command": self.command,
            "started": self.started,
            "finished": self.finished,
            "returncode": self.returncode,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "output": self.output,
            "error": self.error,
        }


STATE = JobState()
LOCK = threading.Lock()
FLASH_MESSAGE = ""
JOB_PROCESS: subprocess.Popen[str] | None = None


def _run_command(command: list[str]) -> None:
    global JOB_PROCESS
    env = os.environ.copy()
    env["2DVIDEO_PROGRESS_FILE"] = str(UI_PROGRESS_FILE)
    UI_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    UI_PROGRESS_FILE.write_text(json.dumps({"percent": 0, "stage": "starting", "message": "starting", "status": "running"}) + "\n", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with LOCK:
        STATE.status = "running"
        STATE.title = Path(command[1]).stem if len(command) > 1 else "job"
        STATE.command = command
        STATE.started = time.time()
        STATE.finished = None
        STATE.returncode = None
        STATE.progress = 0.0
        STATE.stage = "starting"
        STATE.message = "starting"
        STATE.output = ""
        STATE.error = ""
        JOB_PROCESS = proc
    stdout, stderr = proc.communicate()
    with LOCK:
        STATE.finished = time.time()
        STATE.returncode = proc.returncode
        STATE.output = stdout or ""
        STATE.error = stderr or ""
        STATE.status = "done" if proc.returncode == 0 else "failed"
        STATE.progress = 100.0 if proc.returncode == 0 else STATE.progress
        if proc.returncode != 0 and STATE.progress is None:
            STATE.progress = 0.0
        JOB_PROCESS = None
    try:
        UI_PROGRESS_FILE.write_text(
            json.dumps(
                {
                    "percent": 100 if proc.returncode == 0 else (STATE.progress or 0),
                    "stage": "complete" if proc.returncode == 0 else "failed",
                    "message": "done" if proc.returncode == 0 else "failed",
                    "status": "done" if proc.returncode == 0 else "failed",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def start_job(command: list[str], title: str) -> None:
    with LOCK:
        if STATE.status == "running":
            raise RuntimeError("A job is already running. Wait for it to finish or stop it first.")
        STATE.title = title
    thread = threading.Thread(target=_run_command, args=(command,), daemon=True)
    thread.start()


def stop_job() -> None:
    global JOB_PROCESS
    with LOCK:
        proc = JOB_PROCESS
        if proc is None:
            return
        JOB_PROCESS = None
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    with LOCK:
        STATE.status = "stopped"
        STATE.finished = time.time()
        STATE.progress = 0.0
        STATE.stage = "stopped"
        STATE.message = "stopped"


def _latest_output_dirs() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    candidates = [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name != "final"]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[:8]


def _latest_episode_dir() -> Path | None:
    if not OUTPUT_DIR.exists():
        return None
    candidates = [p.parent for p in OUTPUT_DIR.rglob("episode.json") if p.parent.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _episode_dirs(limit: int = 20) -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    candidates = [p.parent for p in OUTPUT_DIR.rglob("episode.json") if p.parent.is_dir()]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
        if len(unique) >= limit:
            break
    return unique


def _episode_label(episode_dir: Path) -> str:
    episode = _read_json(episode_dir / "episode.json") or {}
    episode_id = episode.get("episode_id", episode_dir.name)
    title = episode.get("title")
    if title and title != episode_id:
        return f"{episode_id} · {title}"
    return str(episode_id)


def _resolve_episode_dir(raw: str | None) -> Path | None:
    if raw:
        try:
            candidate = _safe_path(raw)
        except Exception:
            candidate = None
        if candidate is not None:
            if candidate.is_file() and candidate.name == "episode.json":
                return candidate.parent
            if candidate.is_dir() and (candidate / "episode.json").exists():
                return candidate
    return _latest_episode_dir()


def _shared_character_catalog() -> list[dict[str, Any]]:
    path = ROOT / "config" / "shared_character_catalog.json"
    data = _read_json(path)
    return data if isinstance(data, list) else []


def _prompt_environment_names() -> list[str]:
    path = ROOT / "prompt_bibles" / "Fruit_Environment_Bible.md"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    names = []
    for match in re.finditer(r"\|\s*File name\s*\|\s*([^|\n]+)\s*\|", text):
        name = match.group(1).strip()
        if name and name not in names:
            names.append(name)
    return names


def _run_prompt_generator(params: dict[str, str]) -> str:
    mode = params.get("mode", "character")
    command = [sys.executable, str(ROOT / "tools" / "prompt_generator.py")]
    if params.get("use_ollama") == "yes" and mode in {"scene", "video"}:
        command.append("--use-ollama")
    command.append(mode)
    if mode in {"character", "environment"}:
        command.append(params.get("name", ""))
    elif mode == "scene":
        command += [
            "--characters",
            params.get("characters", ""),
            "--environment",
            params.get("environment", ""),
            "--shot",
            params.get("shot", "medium"),
            "--moment",
            params.get("moment", ""),
        ]
    elif mode == "video":
        command += [
            "--characters",
            params.get("characters", ""),
            "--environment",
            params.get("environment", ""),
            "--speaker1",
            params.get("speaker1", ""),
            "--line1",
            params.get("line1", ""),
            "--emotion1",
            params.get("emotion1", ""),
            "--bgm",
            params.get("bgm", ""),
            "--mood",
            params.get("mood", ""),
            "--ambient",
            params.get("ambient", "subtle village ambience"),
            "--camera",
            params.get("camera", "gentle cinematic camera move"),
        ]
        if params.get("speaker2"):
            command += [
                "--speaker2",
                params.get("speaker2", ""),
                "--line2",
                params.get("line2", ""),
                "--emotion2",
                params.get("emotion2", ""),
            ]
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Prompt generation failed.").strip())
    return proc.stdout.strip()


def _episode_meta_path(episode_dir: Path) -> Path:
    return episode_dir / "story.meta.json"


def _read_episode_meta(episode_dir: Path) -> dict[str, Any]:
    data = _read_json(_episode_meta_path(episode_dir))
    return data if isinstance(data, dict) else {"status": "draft", "notes": ""}


def _write_episode_meta(episode_dir: Path, *, status: str, notes: str) -> None:
    write_json(_episode_meta_path(episode_dir), {"status": status, "notes": notes, "updated": time.time()})


def _character_preview_asset(entry: dict[str, Any]) -> Path | None:
    character_id = str(entry.get("character_id", ""))
    shared_preview = ROOT / "assets" / "shared_character_portraits" / f"{character_id}.png"
    if shared_preview.exists():
        return shared_preview
    if character_id in {"char_01_girl", "char_01_bunny", "char_02_fox", "char_03_bear", "char_04_cat", "char_05_owl"}:
        return _character_preview_path(character_id)
    return None


def _safe_path(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (ROOT / raw).resolve()
    else:
        candidate = candidate.resolve()
    root = ROOT.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Invalid path")
    return candidate


def _read_text(path: Path, limit: int = 5000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - display helper
        return f"<unable to read {path}: {exc}>"
    if len(text) > limit:
        return text[:limit] + "\n... truncated ..."
    return text


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _wav_duration_sec(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as fh:
            frames = fh.getnframes()
            rate = fh.getframerate()
            return frames / max(rate, 1)
    except Exception:
        return 0.0


def _waveform_bars(seed: int, count: int = 12) -> list[int]:
    bars: list[int] = []
    value = max(1, seed)
    for idx in range(count):
        value = (value * 1103515245 + 12345 + idx * 97) & 0x7FFFFFFF
        bars.append(24 + (value % 76))
    return bars


def _story_preview_url(path: Path) -> str:
    return _rel_file_url(path)


def _safe_js_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _preview_modal_markup() -> str:
    return """
    <div id="previewModal" class="preview-modal" aria-hidden="true">
      <div class="preview-backdrop" onclick="closePreview()"></div>
      <div class="preview-dialog" role="dialog" aria-modal="true" aria-label="Image preview">
        <button type="button" class="preview-close" onclick="closePreview()">Close</button>
        <div id="previewTitle" class="preview-title"></div>
        <img id="previewImage" alt="" />
      </div>
    </div>
    <script>
      function openPreview(src, title) {
        const modal = document.getElementById('previewModal');
        const img = document.getElementById('previewImage');
        const label = document.getElementById('previewTitle');
        img.src = src;
        img.alt = title || 'Preview';
        label.textContent = title || '';
        modal.setAttribute('aria-hidden', 'false');
        modal.classList.add('open');
      }
      function closePreview() {
        const modal = document.getElementById('previewModal');
        const img = document.getElementById('previewImage');
        modal.classList.remove('open');
        modal.setAttribute('aria-hidden', 'true');
        img.src = '';
      }
      document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
          closePreview();
        }
      });
      function seekAudio(audioId, seconds) {
        const audio = document.getElementById(audioId);
        if (!audio) return;
        audio.currentTime = seconds;
        audio.play();
      }
    </script>
    """


def _episode_artifacts(episode_dir: Path) -> list[tuple[str, Path]]:
    candidates = [
        ("episode.json", episode_dir / "episode.json"),
        ("validated", episode_dir / "episode.validated.json"),
        ("tts", episode_dir / "episode.tts.json"),
        ("rendered", episode_dir / "episode.rendered.json"),
        ("broll", episode_dir / "episode.broll.json"),
        ("assembled manifest", episode_dir / "assembled.manifest.json"),
        ("final video", episode_dir / "final.mp4"),
        ("assembled video", episode_dir / "assembled.mp4"),
        ("audio", episode_dir / "assembled.wav"),
    ]
    return [(label, path) for label, path in candidates if path.exists()]


def _rel_file_url(path: Path) -> str:
    return f"/file?path={quote(str(path.relative_to(ROOT)))}"


def _first_existing(path_candidates: list[Path]) -> Path | None:
    for candidate in path_candidates:
        if candidate.exists():
            return candidate
    return None


def _character_preview_path(character_id: str) -> Path | None:
    char_dir = CHAR_DIR / character_id
    if not char_dir.exists():
        return None
    return _first_existing(
        [
            char_dir / "source_sheet.png",
            char_dir / "body.png",
            char_dir / "head.png",
            char_dir / "face_neutral.png",
            char_dir / "face_smile.png",
            char_dir / "eyes_open.png",
            char_dir / "mouth_closed.png",
        ]
    )


def _background_preview_path(background: str) -> Path:
    path = ROOT / "assets" / "backgrounds" / f"{background}.png"
    if path.exists():
        return path
    return generate_background(background)


def _shot_audio_items(episode: dict[str, Any], episode_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for shot in episode.get("shots", []):
        for index, line in enumerate(shot.get("dialogue", []), start=1):
            audio_path = line.get("audio_path")
            resolved = None
            exists = False
            duration_sec = 0.0
            timings = line.get("timings", []) or []
            if audio_path:
                resolved = Path(audio_path)
                if not resolved.is_absolute():
                    resolved = (episode_dir / resolved).resolve()
                exists = resolved.exists()
                if exists:
                    duration_sec = _wav_duration_sec(resolved)
            if not duration_sec and timings:
                try:
                    duration_sec = float(timings[-1].get("end_sec", 0.0))
                except Exception:
                    duration_sec = 0.0
            seed = sum(ord(ch) for ch in f"{shot.get('shot_id', '')}:{line.get('character', '')}:{line.get('line', '')}")
            waveform = _waveform_bars(seed)
            items.append(
                {
                    "shot_id": shot.get("shot_id", ""),
                    "line_index": index,
                    "character": line.get("character", ""),
                    "voice_id": line.get("voice_id", ""),
                    "audio_path": audio_path or "",
                    "resolved_audio_path": str(resolved) if resolved else "",
                    "exists": exists,
                    "text": line.get("line", ""),
                    "duration_sec": round(duration_sec, 2) if duration_sec else 0.0,
                    "duration_label": f"{duration_sec:.1f}s" if duration_sec else "n/a",
                    "waveform": waveform,
                    "timings": timings,
                }
            )
    return items


def _story_status(episode_dir: Path) -> dict[str, Any]:
    episode_json = _read_json(episode_dir / "episode.json") or {}
    tts_episode = _read_json(episode_dir / "episode.tts.json") or episode_json
    render_episode = _read_json(episode_dir / "episode.rendered.json") or episode_json
    broll_episode = _read_json(episode_dir / "episode.broll.json") or episode_json
    mix_json = _read_json(episode_dir / "assembled.manifest.json") or {}
    qa_json = _read_json(episode_dir / "final.qa_report.json") or {}
    audio_items = _shot_audio_items(tts_episode if isinstance(tts_episode, dict) else episode_json, episode_dir)
    audio_missing = [item for item in audio_items if item["audio_path"] and not item["exists"]]
    final_mp4 = episode_dir / "final.mp4"
    assembled_mp4 = episode_dir / "assembled.mp4"
    assembled_wav = episode_dir / "assembled.wav"
    return {
        "episode": tts_episode if isinstance(tts_episode, dict) else episode_json,
        "tts_count": len((tts_episode if isinstance(tts_episode, dict) else episode_json).get("tts_manifest", [])),
        "render_count": len((render_episode if isinstance(render_episode, dict) else episode_json).get("render_manifest", [])),
        "broll_count": len((broll_episode if isinstance(broll_episode, dict) else episode_json).get("broll_manifest", [])),
        "audio_items": audio_items,
        "audio_missing_count": len(audio_missing),
        "audio_mix_status": "missing" if not audio_items else ("partial" if audio_missing else "ready"),
        "assembled_wav": assembled_wav,
        "assembled_mp4": assembled_mp4,
        "final_mp4": final_mp4,
        "mix_json": mix_json,
        "qa_json": qa_json,
        "final_status": "done" if final_mp4.exists() else ("assembling" if assembled_mp4.exists() or assembled_wav.exists() else "pending"),
    }


def _storyboard_cards(episode_dir: Path, episode_json: dict[str, Any]) -> str:
    cards: list[str] = []
    registry = _read_json(CHAR_DIR / "registry.json") or {}
    for shot in episode_json.get("shots", []):
        bg_name = str(shot.get("background", "meadow_day"))
        bg_path = _background_preview_path(bg_name)
        char_cards = []
        for character_id in shot.get("characters", []):
            char_entry = registry.get(character_id, {})
            preview = _character_preview_path(character_id)
            if preview is not None:
                char_cards.append(
                    f"""
                    <div class="asset-chip">
                      <button type="button" class="preview-trigger" onclick="openPreview('{_safe_js_string(_rel_file_url(preview))}', '{_safe_js_string(html.escape(str(char_entry.get('name', character_id))))}')">
                        <img src="{_rel_file_url(preview)}" alt="{html.escape(character_id)}" />
                      </button>
                      <div>
                        <strong>{html.escape(str(char_entry.get('name', character_id)))}</strong>
                        <span>{html.escape(character_id)}</span>
                      </div>
                    </div>
                    """
                )
            else:
                char_cards.append(
                    f"""
                    <div class="asset-chip empty">
                      <div>
                        <strong>{html.escape(str(char_entry.get('name', character_id)) or character_id)}</strong>
                        <span>{html.escape(character_id)} preview unavailable</span>
                      </div>
                    </div>
                    """
                )
        if not char_cards:
            char_cards.append('<div class="asset-chip empty"><div><strong>B-roll</strong><span>No characters in frame</span></div></div>')
        dialogue_rows = []
        for item in _shot_audio_items({"shots": [shot]}, episode_dir):
            audio_state = "available" if item["exists"] else ("missing" if item["audio_path"] else "not set")
            audio_class = audio_state.replace(" ", "-")
            audio_link = ""
            if item["exists"] and item["resolved_audio_path"]:
                audio_link = f' <a href="{_rel_file_url(Path(item["resolved_audio_path"]))}" target="_blank" rel="noopener">audio</a>'
            waveform_html = "".join(
                f'<span style="height:{height}%" title="{height}%"></span>' for height in item["waveform"]
            )
            dialogue_rows.append(
                f"""
                <li>
                  <div class="dialogue-line">
                  <div>
                    <strong>{html.escape(item['character'] or 'Unknown')}</strong>
                      <span class="audio-badge {html.escape(audio_class)}">{html.escape(audio_state)}</span>
                      <span class="audio-badge duration">{html.escape(item['duration_label'])}</span>
                      {audio_link}
                    </div>
                    <div class="waveform">{waveform_html}</div>
                  </div>
                  <span>{html.escape(item['text'])}</span>
                </li>
                """
            )
        cards.append(
            f"""
            <article class="story-card">
              <div class="story-media">
                <div class="media-label">Background: {html.escape(bg_name)}</div>
                <button type="button" class="preview-trigger preview-full" onclick="openPreview('{_safe_js_string(_rel_file_url(bg_path))}', '{_safe_js_string(bg_name)}')">
                  <img src="{_rel_file_url(bg_path)}" alt="{html.escape(bg_name)} background" />
                </button>
              </div>
              <div class="story-body">
                <div class="story-head">
                  <div>
                    <h3>{html.escape(str(shot.get('shot_id', '')))}</h3>
                    <p>{html.escape(str(shot.get('duration_sec', '')))} sec · {html.escape('B-roll' if shot.get('broll') else 'Character scene')}</p>
                  </div>
                  <div class="pill small">{html.escape('B-roll' if shot.get('broll') else 'Dialogue')}</div>
                </div>
                <div class="asset-stack">{''.join(char_cards)}</div>
                <p class="shot-action">{html.escape(str(shot.get('action', shot.get('video_prompt', '')) or ''))}</p>
                <ul class="dialogue-list">{''.join(dialogue_rows) if dialogue_rows else '<li>No dialogue lines.</li>'}</ul>
              </div>
            </article>
            """
        )
    return "".join(cards) if cards else '<p class="muted">No story shots found.</p>'


def _audio_mix_report_html(episode_dir: Path, story_status: dict[str, Any]) -> str:
    blocks: list[str] = []
    for item in story_status.get("audio_items", []):
        audio_state = "available" if item["exists"] else "missing"
        audio_id = f"audio-{item['shot_id']}-{item['line_index']}".replace(" ", "-")
        audio_player = ""
        markers_html = ""
        if item["exists"] and item["resolved_audio_path"]:
            audio_player = (
                f'<audio id="{html.escape(audio_id)}" controls preload="none" src="{_rel_file_url(Path(item["resolved_audio_path"]))}"></audio>'
            )
            markers: list[str] = []
            timings = item.get("timings", [])
            for idx, timing in enumerate(timings, start=1):
                label = str(timing.get("word", idx))
                start = float(timing.get("start_sec", 0.0) or 0.0)
                markers.append(
                    f'<button type="button" class="timing-marker" onclick="seekAudio(\'{_safe_js_string(audio_id)}\', {start:.3f})">{html.escape(label)}</button>'
                )
            markers_html = f'<div class="timing-strip">{"".join(markers) if markers else "<span class=\"muted\">No timings</span>"}</div>'
        blocks.append(
            f"""
            <article class="audio-line-card">
              <div class="audio-line-top">
                <div>
                  <strong>{html.escape(item["shot_id"])} / line {item["line_index"]}</strong>
                  <span class="audio-badge {audio_state}">{html.escape(audio_state)}</span>
                  <span class="audio-badge duration">{html.escape(item["duration_label"])}</span>
                </div>
                <div class="waveform">{''.join(f'<span style="height:{height}%"></span>' for height in item["waveform"])}</div>
              </div>
              <p>{html.escape(item["character"] or "Unknown")} · {html.escape(item["text"])}</p>
              {audio_player}
              {markers_html}
            </article>
            """
        )
    return "".join(blocks) if blocks else '<p class="muted">No audio lines found yet.</p>'


def _render_character_library_page() -> str:
    shared = _shared_character_catalog()
    registry = _read_json(CHAR_DIR / "registry.json") or {}
    cards: list[str] = []
    for entry in shared:
        character_id = str(entry.get("character_id", ""))
        voice_id = str(entry.get("voice_id", character_id))
        imported = next((item for item in registry.values() if item.get("name", "").lower() == str(entry.get("name", "")).lower()), None)
        preview = _character_preview_asset(entry)
        preview_url = _rel_file_url(preview) if preview else ""
        sample_path = ROOT / "config" / "voices" / f"{voice_id}_sample.wav"
        sample_html = ""
        if sample_path.exists():
            sample_html = (
                f'<div class="voice-sample"><div class="voice-sample-head"><strong>Voice sample</strong><span>{html.escape(voice_id)}</span></div>'
                f'<audio controls preload="none" src="{_rel_file_url(sample_path)}"></audio></div>'
            )
        status = "imported" if imported or preview else "library only"
        card_image = f'<button type="button" class="preview-trigger preview-full" onclick="openPreview(\'{_safe_js_string(preview_url)}\', \'{_safe_js_string(str(entry.get("name", "")))}\')"><img src="{preview_url}" alt="{html.escape(str(entry.get("name", "")))}" /></button>' if preview else f'<div class="library-placeholder">{html.escape(str(entry.get("file_name", "")))}</div>'
        cards.append(
            f"""
            <article class="library-card" data-search="{html.escape(' '.join([
                str(entry.get('name', '')),
                str(entry.get('fruit_type', '')),
                str(entry.get('file_name', '')),
                voice_id,
                str(entry.get('personality', '')),
                str(entry.get('voice', '')),
                str(entry.get('acting', '')),
            ]).lower())}">
              <div class="library-art">{card_image}</div>
              <div class="library-body">
                <div class="library-head">
                  <div>
                    <h3>{html.escape(str(entry.get("name", "")))}</h3>
                    <p>{html.escape(str(entry.get("fruit_type", "")))} · {html.escape(str(entry.get("file_name", "")))}</p>
                  </div>
                  <span class="pill small">{html.escape(status)}</span>
                </div>
                <p><strong>Personality:</strong> {html.escape(str(entry.get("personality", "")))}</p>
                <p><strong>Voice:</strong> {html.escape(str(entry.get("voice", "")))}</p>
                {sample_html}
                <p><strong>Acting:</strong> {html.escape(str(entry.get("acting", "")))}</p>
                <pre class="library-prompt">{html.escape(str(entry.get("image_prompt", "")))}</pre>
              </div>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Characters</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: linear-gradient(135deg, #eef7f5, #f8efe0); color: #1e293b; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 28px 18px 60px; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 18px; }}
    .topbar a {{ color: white; background: #0f766e; text-decoration: none; padding: 10px 14px; border-radius: 999px; font-weight: 800; }}
    .toolbar {{ display:grid; gap:12px; grid-template-columns: minmax(0, 1fr) auto; align-items:end; margin: 16px 0 20px; }}
    .toolbar input {{ border-radius: 14px; border: 1px solid rgba(30,41,59,.12); padding: 11px 12px; font: inherit; width: 100%; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .library-card {{ display: grid; grid-template-columns: 240px minmax(0, 1fr); background: rgba(255,255,255,.88); border: 1px solid rgba(30,41,59,.12); border-radius: 20px; overflow: hidden; box-shadow: 0 18px 40px rgba(15,23,42,.12); }}
    .library-art {{ min-height: 100%; background: rgba(15,23,42,.04); }}
    .library-art img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .library-placeholder {{ display:flex; align-items:center; justify-content:center; min-height: 100%; color:#64748b; font-weight:800; }}
    .library-body {{ padding: 16px; display: grid; gap: 8px; align-content: start; }}
    .library-head {{ display:flex; justify-content: space-between; gap: 10px; align-items:start; }}
    .library-head h3 {{ margin: 0; }}
    .library-head p {{ margin: 4px 0 0; color: #64748b; }}
    .library-prompt {{ white-space: pre-wrap; background: rgba(15,23,42,.05); border-radius: 14px; padding: 12px; margin: 0; }}
    .voice-sample {{ display: grid; gap: 8px; padding: 10px 12px; border-radius: 14px; background: rgba(15,118,110,.08); border: 1px solid rgba(15,118,110,.14); }}
    .voice-sample-head {{ display:flex; justify-content: space-between; gap: 10px; align-items: center; font-size: 0.85rem; color: #0f766e; }}
    .voice-sample audio {{ width: 100%; }}
    .preview-modal {{ position: fixed; inset: 0; display: none; align-items: center; justify-content: center; z-index: 1000; }}
    .preview-modal.open {{ display: flex; }}
    .preview-backdrop {{ position:absolute; inset:0; background: rgba(15,23,42,.78); backdrop-filter: blur(4px); }}
    .preview-dialog {{ position:relative; z-index:1; max-width:min(92vw,1100px); max-height:90vh; background:white; border-radius:20px; box-shadow:0 30px 90px rgba(0,0,0,.35); padding:16px; display:grid; gap:10px; }}
    .preview-dialog img {{ max-width:100%; max-height:78vh; border-radius:16px; object-fit:contain; background:#f8fafc; }}
    .preview-title {{ font-weight:800; }}
    .preview-close {{ justify-self:end; border:0; background:#0f766e; color:white; padding:10px 14px; border-radius:999px; font-weight:800; }}
    .hidden {{ display:none !important; }}
    @media (max-width: 900px) {{ .grid, .library-card {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>Character Library</h1>
        <p>All shared characters from the bible, plus any imported cast in the workspace.</p>
      </div>
      <a href="/">Back to UI</a>
    </div>
    <div class="toolbar">
      <label>Search characters
        <input id="characterSearch" type="search" placeholder="Search name, fruit, file, personality, voice..." oninput="filterCharacters()" />
      </label>
      <div class="status-chip" id="characterCount">{len(cards) if cards else 0} characters</div>
    </div>
    <div class="grid">
      <div id="characterGrid" style="grid-column:1 / -1; display:grid; gap:18px; grid-template-columns: repeat(2, minmax(0, 1fr));">
        {''.join(cards) if cards else '<p>No shared characters found.</p>'}
      </div>
    </div>
  </div>
  {_preview_modal_markup()}
  <script>
    function filterCharacters() {{
      const query = document.getElementById('characterSearch').value.trim().toLowerCase();
      const cards = document.querySelectorAll('#characterGrid .library-card');
      let visible = 0;
      cards.forEach(card => {{
        const haystack = card.getAttribute('data-search') || '';
        const match = haystack.includes(query);
        card.classList.toggle('hidden', !match);
        if (match) visible += 1;
      }});
      document.getElementById('characterCount').textContent = visible + ' characters';
    }}
  </script>
</body>
</html>"""


def _render_prompt_generator_page(query: dict[str, list[str]]) -> str:
    shared = _shared_character_catalog()
    character_names = [str(entry.get("name", "")) for entry in shared if entry.get("name")]
    environment_names = _prompt_environment_names()
    params = {key: values[0] for key, values in query.items() if values}
    mode = params.get("mode", "character")
    if "name" not in params:
        params["name"] = character_names[0] if mode == "character" and character_names else (environment_names[0] if environment_names else "")
    if "characters" not in params:
        params["characters"] = ",".join(character_names[:2]) if len(character_names) >= 2 else ",".join(character_names)
    if "environment" not in params:
        params["environment"] = environment_names[0] if environment_names else ""
    output = ""
    error = ""
    if params.get("generate") == "yes":
        try:
            output = _run_prompt_generator(params)
        except Exception as exc:
            error = str(exc)

    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    def options(values: list[str], selected: str) -> str:
        return "".join(f'<option value="{esc(value)}"{" selected" if value == selected else ""}>{esc(value)}</option>' for value in values)

    character_options = options(character_names, params.get("name", ""))
    speaker_options = options(character_names, params.get("speaker1", character_names[0] if character_names else ""))
    speaker2_options = '<option value="">None</option>' + options(character_names, params.get("speaker2", ""))
    environment_options = options(environment_names, params.get("environment", ""))
    use_ollama_checked = " checked" if params.get("use_ollama") == "yes" else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt Generator</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#f6f8f3; color:#172033; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 26px 18px 60px; }}
    .topbar {{ display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:18px; }}
    .topbar nav {{ display:flex; gap:8px; flex-wrap:wrap; }}
    a, button {{ border:0; border-radius:8px; padding:10px 12px; background:#0f766e; color:white; text-decoration:none; font-weight:800; cursor:pointer; }}
    h1 {{ margin:0; font-size:2rem; }}
    p {{ color:#5b6472; }}
    .layout {{ display:grid; grid-template-columns: 390px minmax(0,1fr); gap:18px; align-items:start; }}
    form {{ display:grid; gap:12px; background:white; border:1px solid rgba(23,32,51,.12); border-radius:8px; padding:16px; box-shadow:0 12px 30px rgba(23,32,51,.08); }}
    label {{ display:grid; gap:6px; font-size:.9rem; font-weight:800; }}
    input, select, textarea {{ border:1px solid rgba(23,32,51,.16); border-radius:8px; padding:10px 11px; font:inherit; }}
    textarea {{ min-height:84px; resize:vertical; }}
    .inline {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; }}
    .check {{ display:flex; align-items:center; gap:8px; font-weight:800; }}
    .check input {{ width:auto; }}
    .panel {{ background:white; border:1px solid rgba(23,32,51,.12); border-radius:8px; padding:16px; box-shadow:0 12px 30px rgba(23,32,51,.08); }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#101827; color:#eff6ff; border-radius:8px; padding:16px; min-height:320px; margin:0; line-height:1.55; }}
    .error {{ background:#fee2e2; color:#991b1b; border-radius:8px; padding:12px; font-weight:700; }}
    @media (max-width: 900px) {{ .layout, .inline {{ grid-template-columns:1fr; }} .topbar {{ align-items:flex-start; flex-direction:column; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>Prompt Generator</h1>
        <p>Generate locked prompts from the fruit bibles. Ollama is optional.</p>
      </div>
      <nav><a href="/">Dashboard</a><a href="/characters">Characters</a><a href="/stories">Stories</a></nav>
    </div>
    <div class="layout">
      <form method="get" action="/prompts">
        <input type="hidden" name="generate" value="yes" />
        <label>Mode
          <select name="mode" onchange="this.form.submit()">
            <option value="character"{" selected" if mode == "character" else ""}>Character</option>
            <option value="environment"{" selected" if mode == "environment" else ""}>Environment</option>
            <option value="scene"{" selected" if mode == "scene" else ""}>Scene</option>
            <option value="video"{" selected" if mode == "video" else ""}>Video</option>
          </select>
        </label>
        {f'''<label>Character
          <select name="name">{character_options}</select>
        </label>''' if mode == "character" else ""}
        {f'''<label>Environment
          <select name="name">{environment_options}</select>
        </label>''' if mode == "environment" else ""}
        {f'''<label>Characters
          <input name="characters" value="{esc(params.get("characters", ""))}" />
        </label>
        <label>Environment
          <select name="environment">{environment_options}</select>
        </label>
        <div class="inline">
          <label>Shot
            <input name="shot" value="{esc(params.get("shot", "close-up"))}" />
          </label>
          <label>Moment
            <input name="moment" value="{esc(params.get("moment", "Appy realizes Ozzy lied"))}" />
          </label>
        </div>''' if mode == "scene" else ""}
        {f'''<label>Characters
          <input name="characters" value="{esc(params.get("characters", ""))}" />
        </label>
        <label>Environment
          <select name="environment">{environment_options}</select>
        </label>
        <div class="inline">
          <label>Speaker 1
            <select name="speaker1">{speaker_options}</select>
          </label>
          <label>Speaker 2
            <select name="speaker2">{speaker2_options}</select>
          </label>
        </div>
        <label>Line 1
          <input name="line1" value="{esc(params.get("line1", "Tumne mujhse jhoot kyun bola?"))}" />
        </label>
        <label>Emotion 1
          <input name="emotion1" value="{esc(params.get("emotion1", "hurt, quietly betrayed"))}" />
        </label>
        <label>Line 2
          <input name="line2" value="{esc(params.get("line2", ""))}" />
        </label>
        <label>Emotion 2
          <input name="emotion2" value="{esc(params.get("emotion2", ""))}" />
        </label>
        <div class="inline">
          <label>BGM
            <input name="bgm" value="{esc(params.get("bgm", "Betrayal reveal"))}" />
          </label>
          <label>Mood
            <input name="mood" value="{esc(params.get("mood", "heartbreaking"))}" />
          </label>
        </div>
        <label>Ambient
          <input name="ambient" value="{esc(params.get("ambient", "subtle village ambience"))}" />
        </label>
        <label>Camera
          <input name="camera" value="{esc(params.get("camera", "gentle cinematic camera move"))}" />
        </label>''' if mode == "video" else ""}
        <label class="check"><input type="checkbox" name="use_ollama" value="yes"{use_ollama_checked} /> Use Ollama</label>
        <button type="submit">Generate Prompt</button>
      </form>
      <div class="panel">
        {f'<div class="error">{esc(error)}</div>' if error else ''}
        <pre>{esc(output or "Choose settings and generate a prompt.")}</pre>
      </div>
    </div>
  </div>
</body>
</html>"""


def _compare_episode_data(left: Path, right: Path) -> str:
    left_status = _story_status(left)
    right_status = _story_status(right)
    left_episode = left_status["episode"]
    right_episode = right_status["episode"]
    left_bg = left_episode.get("shots", [{}])[0].get("background", "meadow_day") if left_episode.get("shots") else "meadow_day"
    right_bg = right_episode.get("shots", [{}])[0].get("background", "meadow_day") if right_episode.get("shots") else "meadow_day"
    left_bg_url = _rel_file_url(_background_preview_path(str(left_bg)))
    right_bg_url = _rel_file_url(_background_preview_path(str(right_bg)))
    return f"""
    <div class="compare-col">
      <div class="compare-media"><img src="{left_bg_url}" alt="Left cover" /></div>
      <h3>{html.escape(_episode_label(left))}</h3>
      <p>{html.escape(left_episode.get("type", ""))} · {len(left_episode.get("shots", []))} shots</p>
      <p>Status: {html.escape(left_status["final_status"])} · Audio: {html.escape(left_status["audio_mix_status"])}</p>
      <p>Missing audio: {html.escape(str(left_status["audio_missing_count"]))}</p>
    </div>
    <div class="compare-col">
      <div class="compare-media"><img src="{right_bg_url}" alt="Right cover" /></div>
      <h3>{html.escape(_episode_label(right))}</h3>
      <p>{html.escape(right_episode.get("type", ""))} · {len(right_episode.get("shots", []))} shots</p>
      <p>Status: {html.escape(right_status["final_status"])} · Audio: {html.escape(right_status["audio_mix_status"])}</p>
      <p>Missing audio: {html.escape(str(right_status["audio_missing_count"]))}</p>
    </div>
    """


def _compare_shot_diff_html(left: Path, right: Path) -> str:
    left_episode = _story_status(left)["episode"]
    right_episode = _story_status(right)["episode"]
    left_shots = left_episode.get("shots", [])
    right_shots = right_episode.get("shots", [])
    max_len = max(len(left_shots), len(right_shots))
    rows: list[str] = []
    for idx in range(max_len):
        left_shot = left_shots[idx] if idx < len(left_shots) else {}
        right_shot = right_shots[idx] if idx < len(right_shots) else {}
        left_bg = str(left_shot.get("background", ""))
        right_bg = str(right_shot.get("background", ""))
        bg_change = "same" if left_bg == right_bg else "changed"
        left_chars = ", ".join(left_shot.get("characters", [])) or "broll"
        right_chars = ", ".join(right_shot.get("characters", [])) or "broll"
        rows.append(
            f"""
            <tr>
              <td>{idx + 1}</td>
              <td>{html.escape(str(left_shot.get("shot_id", "")))}</td>
              <td>{html.escape(left_bg)}</td>
              <td>{html.escape(left_chars)}</td>
              <td>{html.escape(str(right_shot.get("shot_id", "")))}</td>
              <td>{html.escape(right_bg)}</td>
              <td>{html.escape(right_chars)}</td>
              <td>{html.escape(bg_change)}</td>
            </tr>
            """
        )
    return f"""
    <div class="shot-diff-panel">
      <h2>Shot Diff</h2>
      <table>
        <thead>
          <tr><th>#</th><th>Left Shot</th><th>Left BG</th><th>Left Cast</th><th>Right Shot</th><th>Right BG</th><th>Right Cast</th><th>BG</th></tr>
        </thead>
        <tbody>
          {''.join(rows) if rows else '<tr><td colspan="8">No shots to compare.</td></tr>'}
        </tbody>
      </table>
    </div>
    """


def _render_compare_page(left: Path, right: Path) -> str:
    left_label = _episode_label(left)
    right_label = _episode_label(right)
    left_meta = _read_episode_meta(left)
    right_meta = _read_episode_meta(right)
    left_status = _story_status(left)
    right_status = _story_status(right)
    left_cards = _storyboard_cards(left, left_status["episode"])
    right_cards = _storyboard_cards(right, right_status["episode"])
    shot_diff = _compare_shot_diff_html(left, right)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Compare</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: linear-gradient(135deg, #eef7f5, #f8efe0); color:#1e293b; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 28px 18px 60px; }}
    .topbar {{ display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:18px; }}
    .topbar a {{ color:white; background:#0f766e; text-decoration:none; padding:10px 14px; border-radius:999px; font-weight:800; }}
    .compare-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:18px; }}
    .compare-col {{ background: rgba(255,255,255,.88); border:1px solid rgba(30,41,59,.12); border-radius:20px; overflow:hidden; box-shadow:0 18px 40px rgba(15,23,42,.12); padding: 16px; }}
    .compare-media img {{ width:100%; height:260px; object-fit:cover; border-radius:16px; display:block; margin-bottom:12px; }}
    .compare-status {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:12px; margin: 16px 0; }}
    .metric {{ border:1px solid rgba(30,41,59,.12); border-radius:16px; padding:12px; background: rgba(255,255,255,.7); }}
    .story-grid {{ margin-top: 16px; }}
    .story-grid .story-card {{ margin-bottom: 16px; }}
    .shot-diff-panel {{ background: rgba(255,255,255,.9); border:1px solid rgba(30,41,59,.12); border-radius:20px; padding:16px; margin: 16px 0; overflow:auto; }}
    .shot-diff-panel table {{ width:100%; border-collapse: collapse; }}
    .shot-diff-panel th, .shot-diff-panel td {{ padding: 8px; border-bottom: 1px solid rgba(30,41,59,.12); text-align:left; vertical-align: top; }}
    .shot-diff-panel th {{ position: sticky; top:0; background: rgba(248,250,252,.96); }}
    .note-panel {{ background: rgba(255,255,255,.9); border:1px solid rgba(30,41,59,.12); border-radius:20px; padding:16px; margin-top:16px; }}
    textarea {{ width:100%; min-height:120px; border-radius:14px; border:1px solid rgba(30,41,59,.12); padding:12px; font:inherit; }}
    select, input {{ width:100%; border-radius:14px; border:1px solid rgba(30,41,59,.12); padding:11px 12px; font:inherit; }}
    .status-row {{ display:flex; gap:12px; flex-wrap:wrap; }}
    .status-chip {{ padding:8px 12px; background:#0f766e; color:white; border-radius:999px; font-weight:800; }}
    @media (max-width: 900px) {{ .compare-grid, .compare-status {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>Compare Episodes</h1>
        <p>Side-by-side story comparison for visuals, status, and notes.</p>
      </div>
      <a href="/">Back to UI</a>
    </div>
    <div class="compare-grid">
      {_compare_episode_data(left, right)}
    </div>
    <div class="compare-status">
      <div class="metric"><strong>{html.escape(left_status["final_status"])}</strong><span>{html.escape(left_label)} final</span></div>
      <div class="metric"><strong>{html.escape(left_status["audio_mix_status"])}</strong><span>{html.escape(left_label)} audio</span></div>
      <div class="metric"><strong>{html.escape(right_status["final_status"])}</strong><span>{html.escape(right_label)} final</span></div>
      <div class="metric"><strong>{html.escape(right_status["audio_mix_status"])}</strong><span>{html.escape(right_label)} audio</span></div>
    </div>
    <div class="note-panel">
      <h2>Left Notes</h2>
      <pre>{html.escape(json.dumps(left_meta, indent=2, ensure_ascii=True))}</pre>
      <h2>Right Notes</h2>
      <pre>{html.escape(json.dumps(right_meta, indent=2, ensure_ascii=True))}</pre>
    </div>
    {shot_diff}
    <div class="story-grid">
      <h2>{html.escape(left_label)}</h2>
      {left_cards}
      <h2>{html.escape(right_label)}</h2>
      {right_cards}
    </div>
  </div>
  {_preview_modal_markup()}
</body>
</html>"""


def _job_summary(progress: dict[str, Any] | None = None) -> str:
    with LOCK:
        state = STATE.as_dict()
    progress = progress or {}
    started = f"{time.ctime(state['started'])}" if state["started"] else "n/a"
    finished = f"{time.ctime(state['finished'])}" if state["finished"] else "n/a"
    progress_status = progress.get("status", "")
    status = state["status"]
    if status == "idle" and progress_status == "running":
        status = "running"
    progress_pct = progress.get("percent", state.get("progress"))
    progress_stage = progress.get("stage", state.get("stage"))
    progress_message = progress.get("message", state.get("message"))
    lines = [
        f"Status: {status}",
        f"Title: {state['title']}",
        f"Command: {' '.join(state['command']) if state['command'] else 'n/a'}",
        f"Started: {started}",
        f"Finished: {finished}",
        f"Return code: {state['returncode']}",
        f"Progress: {progress_pct if progress_pct is not None else 'n/a'}%",
        f"Stage: {progress_stage or 'n/a'}",
        f"Message: {progress_message or 'n/a'}",
    ]
    return "\n".join(lines)


def _render_review_page(episode_dir: Path) -> str:
    episode_json = _read_json(episode_dir / "episode.json") or {}
    summary = _read_json(episode_dir / "assembled.manifest.json") or {}
    final_mp4 = episode_dir / "final.mp4"
    assembled_mp4 = episode_dir / "assembled.mp4"
    assembled_wav = episode_dir / "assembled.wav"
    tts_json = episode_dir / "episode.tts.json"
    rendered_json = episode_dir / "episode.rendered.json"
    broll_json = episode_dir / "episode.broll.json"
    artifacts = _episode_artifacts(episode_dir)

    def link(label: str, path: Path) -> str:
        rel = path.relative_to(ROOT)
        return f'<a href="/file?path={quote(str(rel))}" target="_blank" rel="noopener">{html.escape(label)}</a>'

    shots = episode_json.get("shots", [])
    shot_rows = []
    for shot in shots:
        shot_rows.append(
            "<tr>"
            f"<td>{html.escape(str(shot.get('shot_id', '')))}</td>"
            f"<td>{html.escape(str(shot.get('duration_sec', '')))}</td>"
            f"<td>{html.escape(', '.join(shot.get('characters', [])) or 'broll')}</td>"
            f"<td>{html.escape(str(shot.get('action', 'broll' if shot.get('broll') else '')))}</td>"
            "</tr>"
        )

    summary_json = json.dumps(summary, indent=2, ensure_ascii=True)
    episode_id = episode_json.get("episode_id", episode_dir.name)
    title = episode_json.get("title", episode_id)
    story_status = _story_status(episode_dir)
    story_meta = _read_episode_meta(episode_dir)
    missing_audio = [item for item in story_status["audio_items"] if item["audio_path"] and not item["exists"]]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Review {html.escape(str(episode_id))}</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: linear-gradient(135deg, #eef7f5, #f8efe0);
      color: #1e293b;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .panel {{
      background: rgba(255,255,255,.84);
      border: 1px solid rgba(30,41,59,.12);
      border-radius: 20px;
      box-shadow: 0 16px 36px rgba(15,23,42,.12);
      padding: 20px;
      margin-bottom: 16px;
    }}
    .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .wide {{ grid-column: 1 / -1; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .links a {{
      display: inline-block;
      padding: 10px 12px;
      border-radius: 999px;
      background: #0f766e;
      color: white;
      text-decoration: none;
      font-weight: 700;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px; border-bottom: 1px solid rgba(30,41,59,.12); text-align: left; }}
    video {{ width: 100%; max-height: 520px; border-radius: 16px; background: #000; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: rgba(15,23,42,.05); padding: 14px; border-radius: 16px; overflow-x: auto; }}
    .status-grid {{ display: grid; gap: 10px; grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .metric {{
      border: 1px solid rgba(30,41,59,.12);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,.7);
    }}
    .metric strong {{ display: block; font-size: 1.15rem; margin-bottom: 4px; }}
    .story-grid {{ display: grid; gap: 16px; }}
    .story-card {{
      display: grid;
      gap: 0;
      grid-template-columns: 420px minmax(0, 1fr);
      border: 1px solid rgba(30,41,59,.12);
      border-radius: 18px;
      overflow: hidden;
      background: rgba(255,255,255,.86);
    }}
    .story-media {{ position: relative; min-height: 320px; background: rgba(15,23,42,.04); }}
    .story-media img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .media-label {{
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 1;
      background: rgba(15,23,42,.72);
      color: white;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
    }}
    .story-body {{ padding: 16px; display: grid; gap: 12px; }}
    .story-head {{ display: flex; justify-content: space-between; align-items: start; gap: 12px; }}
    .story-head h3 {{ margin: 0; }}
    .story-head p {{ margin: 4px 0 0; }}
    .pill.small {{ padding: 6px 10px; font-size: 0.8rem; }}
    .asset-stack {{ display: grid; gap: 8px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .asset-chip {{
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 8px;
      border: 1px solid rgba(30,41,59,.12);
      border-radius: 14px;
      background: rgba(248,250,252,.86);
    }}
    .asset-chip img {{
      width: 76px;
      height: 76px;
      object-fit: cover;
      border-radius: 12px;
      background: rgba(15,23,42,.06);
    }}
    .preview-trigger {{
      border: 0;
      background: transparent;
      padding: 0;
      box-shadow: none;
      width: 100%;
    }}
    .preview-trigger img {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .preview-full {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .asset-chip span {{ display: block; color: var(--muted); font-size: 0.82rem; }}
    .asset-chip.empty {{ background: rgba(15,23,42,.03); }}
    .dialogue-list {{ margin: 0; padding-left: 18px; display: grid; gap: 12px; }}
    .dialogue-list li span {{ color: var(--muted); }}
    .shot-action {{ margin: 0; color: var(--muted); }}
    .dialogue-line {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 4px; }}
    .dialogue-line strong {{ margin-right: 8px; }}
    .audio-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 800;
      margin-left: 6px;
      background: rgba(15,23,42,.08);
      color: var(--ink);
    }}
    .audio-badge.available {{ background: rgba(22,163,74,.12); color: #166534; }}
    .audio-badge.missing {{ background: rgba(220,38,38,.12); color: #b91c1c; }}
    .audio-badge.duration {{ background: rgba(37,99,235,.12); color: #1d4ed8; }}
    .waveform {{
      display: flex;
      align-items: end;
      gap: 2px;
      min-width: 90px;
      height: 28px;
      padding: 2px 0;
    }}
    .waveform span {{
      display: block;
      width: 5px;
      border-radius: 999px;
      background: linear-gradient(180deg, #0f766e, #22c55e);
      opacity: 0.86;
    }}
    .audio-line-card {{
      display: grid;
      gap: 10px;
      padding: 14px;
      border: 1px solid rgba(30,41,59,.12);
      border-radius: 16px;
      background: rgba(255,255,255,.78);
    }}
    .audio-line-top {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
    }}
    .audio-line-top > div:first-child {{ line-height: 1.8; }}
    .audio-line-card audio {{ width: 100%; }}
    .timing-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .timing-marker {{
      border: 1px solid rgba(30,41,59,.12);
      background: rgba(15,118,110,.1);
      color: #115e59;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.78rem;
      font-weight: 800;
      box-shadow: none;
    }}
    .missing {{
      border-left: 4px solid #dc2626;
      padding-left: 12px;
      background: rgba(220,38,38,.06);
      border-radius: 12px;
      padding-top: 8px;
      padding-bottom: 8px;
    }}
    .preview-note {{ color: var(--muted); margin-top: 4px; }}
    .preview-modal {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }}
    .preview-modal.open {{ display: flex; }}
    .preview-backdrop {{
      position: absolute;
      inset: 0;
      background: rgba(15,23,42,.78);
      backdrop-filter: blur(4px);
    }}
    .preview-dialog {{
      position: relative;
      z-index: 1;
      max-width: min(92vw, 1100px);
      max-height: 90vh;
      background: white;
      border-radius: 20px;
      box-shadow: 0 30px 90px rgba(0,0,0,.35);
      padding: 16px;
      display: grid;
      gap: 10px;
    }}
    .preview-dialog img {{
      max-width: 100%;
      max-height: 78vh;
      border-radius: 16px;
      object-fit: contain;
      background: #f8fafc;
    }}
    .preview-title {{ font-weight: 800; color: var(--ink); }}
    .preview-close {{
      justify-self: end;
      border: 0;
      background: #0f766e;
      color: white;
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 800;
    }}
    .meta-panel {{
      background: rgba(255,255,255,.9);
      border: 1px solid rgba(30,41,59,.12);
      border-radius: 18px;
      padding: 16px;
    }}
    .meta-panel form {{
      display: grid;
      gap: 10px;
    }}
    .meta-panel textarea {{
      min-height: 120px;
      border-radius: 14px;
      border: 1px solid rgba(30,41,59,.12);
      padding: 12px;
      font: inherit;
    }}
    .meta-panel select {{
      border-radius: 14px;
      border: 1px solid rgba(30,41,59,.12);
      padding: 11px 12px;
      font: inherit;
    }}
    .meta-actions {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
    </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Review: {html.escape(str(title))}</h1>
      <p><strong>Episode:</strong> {html.escape(str(episode_id))}</p>
      <p><strong>Folder:</strong> {html.escape(str(episode_dir.relative_to(ROOT)))}</p>
      <div class="links">
        <a href="/">Back to UI</a>
        <a href="/stories">Stories</a>
        <a href="/characters">Characters</a>
        {link("Final MP4", final_mp4) if final_mp4.exists() else ""}
        {link("Assembled MP4", assembled_mp4) if assembled_mp4.exists() else ""}
        {link("Assembled WAV", assembled_wav) if assembled_wav.exists() else ""}
        {link("Episode JSON", episode_dir / "episode.json") if (episode_dir / "episode.json").exists() else ""}
        {link("TTS JSON", tts_json) if tts_json.exists() else ""}
        {link("Render JSON", rendered_json) if rendered_json.exists() else ""}
        {link("B-roll JSON", broll_json) if broll_json.exists() else ""}
        {link("Manifest", episode_dir / "assembled.manifest.json") if (episode_dir / "assembled.manifest.json").exists() else ""}
      </div>
    </div>
    <div class="grid">
      <div class="panel wide">
        <h2>Pipeline Status</h2>
        <div class="status-grid">
          <div class="metric"><strong>{html.escape(str(story_status["final_status"]))}</strong><span>final output</span></div>
          <div class="metric"><strong>{html.escape(str(story_status["audio_mix_status"]))}</strong><span>audio mix</span></div>
          <div class="metric"><strong>{html.escape(str(story_status["audio_missing_count"]))}</strong><span>missing audio lines</span></div>
          <div class="metric"><strong>{html.escape(str(story_status["tts_count"]))}</strong><span>TTS lines</span></div>
        </div>
        <p class="preview-note">Assembled WAV: {html.escape(str(story_status["assembled_wav"].name)) if story_status["assembled_wav"].exists() else "not found"} · Final MP4: {html.escape(str(story_status["final_mp4"].name)) if story_status["final_mp4"].exists() else "not found"}</p>
      </div>
      <div class="panel wide">
        <h2>Preview Video</h2>
        {f'<video controls src="/file?path={quote(str(final_mp4.relative_to(ROOT)))}"></video>' if final_mp4.exists() else '<p>No final video available.</p>'}
      </div>
      <div class="panel wide">
        <h2>Storyboard</h2>
        <div class="story-grid">
          {_storyboard_cards(episode_dir, episode_json)}
        </div>
      </div>
      <div class="panel wide meta-panel">
        <h2>Story Notes</h2>
        <form method="post" action="/action">
          <input type="hidden" name="action" value="save_episode_meta" />
          <input type="hidden" name="episode_path" value="{html.escape(str(episode_dir.relative_to(ROOT)))}" />
          <label>Status
            <select name="story_status">
              {''.join(f'<option value="{opt}"{" selected" if story_meta.get("status", "draft") == opt else ""}>{opt.title()}</option>' for opt in ["draft", "review", "locked", "published"])}
            </select>
          </label>
          <label>Notes
            <textarea name="notes" placeholder="Write review notes, status comments, or next steps...">{html.escape(str(story_meta.get("notes", "")))}</textarea>
          </label>
          <div class="meta-actions">
            <button type="submit">Save Story Notes</button>
            <a href="/compare?left={quote(str(episode_dir.relative_to(ROOT)))}&right={quote(str((_latest_episode_dir() or episode_dir).relative_to(ROOT)))}">Compare with latest</a>
          </div>
        </form>
      </div>
      <div class="panel">
        <h2>Episode Summary</h2>
        <pre>{html.escape(json.dumps(episode_json, indent=2, ensure_ascii=True)[:8000])}</pre>
      </div>
      <div class="panel">
        <h2>Shot List</h2>
        <table>
          <thead><tr><th>Shot</th><th>Duration</th><th>Characters</th><th>Action</th></tr></thead>
          <tbody>
            {''.join(shot_rows) if shot_rows else '<tr><td colspan="4">No shots found.</td></tr>'}
          </tbody>
        </table>
      </div>
      <div class="panel wide">
        <h2>Audio Mix Report</h2>
        {_audio_mix_report_html(episode_dir, story_status)}
        {f'<p class="missing">Missing audio: {html.escape(str(len(missing_audio)))} line(s) need TTS or have broken paths.</p>' if missing_audio else '<p>All dialogue audio files are present.</p>'}
      </div>
      <div class="panel wide">
        <h2>Assembly Manifest</h2>
        <pre>{html.escape(summary_json[:8000])}</pre>
      </div>
      <div class="panel wide">
        <h2>Available Artifacts</h2>
        <ul>
          {''.join(f'<li>{html.escape(label)} - <a href="/file?path={quote(str(path.relative_to(ROOT)))}" target="_blank" rel="noopener">{html.escape(path.name)}</a></li>' for label, path in artifacts)}
        </ul>
      </div>
    </div>
  </div>
  {_preview_modal_markup()}
</body>
</html>"""


def _read_progress() -> dict[str, Any]:
    try:
        return json.loads(UI_PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _render_page(message: str = "", selected_episode_raw: str | None = None) -> str:
    with LOCK:
        state = STATE.as_dict()
    registry = _read_json(CHAR_DIR / "registry.json") or {}
    voice_registry = _read_json(VOICE_DIR / "voice_registry.json") or {}
    episode_dirs = _episode_dirs()
    latest_dirs = _latest_output_dirs()
    latest_episode = _latest_episode_dir()
    selected_episode = _resolve_episode_dir(selected_episode_raw) or latest_episode
    latest_summary = None
    if latest_dirs:
        summary_path = latest_dirs[0] / "daily_summary.json"
        if summary_path.exists():
            latest_summary = _read_text(summary_path, 10000)
    latest_story_status = _story_status(latest_episode) if latest_episode else None
    latest_storyboard = _storyboard_cards(latest_episode, latest_story_status["episode"]) if latest_story_status else ""
    selected_story_status = _story_status(selected_episode) if selected_episode else None
    selected_storyboard = _storyboard_cards(selected_episode, selected_story_status["episode"]) if selected_story_status else ""
    progress = _read_progress()
    progress_pct = float(progress.get("percent", state.get("progress") or 0.0) or 0.0)
    progress_stage = progress.get("stage", state.get("stage") or "")
    progress_message = progress.get("message", state.get("message") or "")
    auto_refresh = state["status"] == "running" or progress.get("status") == "running"
    jobs_block = _job_summary(progress)

    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    episode_options = []
    for episode_dir in episode_dirs:
        rel = episode_dir.relative_to(ROOT)
        selected_attr = " selected" if selected_episode and selected_episode.resolve() == episode_dir.resolve() else ""
        episode_options.append(f'<option value="{esc(str(rel))}"{selected_attr}>{esc(_episode_label(episode_dir))}</option>')
    selected_label = _episode_label(selected_episode) if selected_episode else "No episode selected"

    latest_cards = []
    for folder in latest_dirs:
        files = sorted([p.name for p in folder.glob("*") if p.is_file()])
        review_link = f'<a href="/review?episode={quote(str(folder.relative_to(ROOT)))}" target="_blank" rel="noopener">Review</a>'
        latest_cards.append(
            f"""
            <div class="card">
              <h3>{esc(folder.name)}</h3>
              <p><strong>Files:</strong> {esc(', '.join(files[:12]))}</p>
              <p>{review_link}</p>
            </div>
            """
        )
    registry_rows = []
    for key, item in registry.items():
        registry_rows.append(
            f"<tr><td>{esc(key)}</td><td>{esc(item.get('name'))}</td><td>{esc(item.get('tier'))}</td><td>{esc(item.get('voice_id'))}</td></tr>"
        )
    voice_rows = []
    for key, item in voice_registry.items():
        voice_rows.append(
            f"<tr><td>{esc(key)}</td><td>{esc(item.get('reference_clip'))}</td><td>{esc(item.get('sample_clip', ''))}</td><td>{esc(item.get('language'))}</td></tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {"<meta http-equiv=\"refresh\" content=\"2\" />" if auto_refresh else ""}
  <title>2DVideo Local UI</title>
  <style>
    :root {{
      --bg: #eef7f5;
      --bg2: #f8efe0;
      --ink: #1e293b;
      --muted: #5b6472;
      --card: rgba(255,255,255,0.82);
      --accent: #0f766e;
      --accent2: #c2410c;
      --line: rgba(30,41,59,.12);
      --shadow: 0 18px 48px rgba(15,23,42,.12);
      --radius: 22px;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,.14), transparent 30%),
        radial-gradient(circle at bottom right, rgba(194,65,12,.14), transparent 28%),
        linear-gradient(135deg, var(--bg), var(--bg2));
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 18px 60px;
    }}
    .hero {{
      display: grid;
      gap: 14px;
      grid-template-columns: 1.4fr .9fr;
      align-items: stretch;
      margin-bottom: 18px;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero .panel, .card, .form-card {{ padding: 20px; }}
    h1 {{ margin: 0; font-size: 2.3rem; line-height: 1; }}
    h2 {{ margin: 0 0 12px; font-size: 1.1rem; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .status {{
      display: grid;
      gap: 8px;
      font-size: 0.95rem;
    }}
    .progress {{
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .progress-bar {{
      width: 100%;
      height: 14px;
      border-radius: 999px;
      background: rgba(30,41,59,.10);
      overflow: hidden;
    }}
    .progress-bar > div {{
      height: 100%;
      width: {progress_pct:.2f}%;
      background: linear-gradient(90deg, var(--accent), #22c55e);
      border-radius: 999px;
      transition: width 0.3s ease;
    }}
    .status-grid {{
      display: grid;
      gap: 10px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(15,118,110,.1);
      color: var(--accent);
      font-weight: 700;
      width: fit-content;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 16px;
    }}
    .wide {{ grid-column: 1 / -1; }}
    .forms {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .form-card form {{
      display: grid;
      gap: 10px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 0.92rem;
      font-weight: 700;
    }}
    input, select, textarea {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 11px 12px;
      font: inherit;
      background: rgba(255,255,255,.92);
      color: var(--ink);
    }}
    textarea {{ min-height: 130px; }}
    button {{
      border: 0;
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, var(--accent), #115e59);
      box-shadow: 0 12px 26px rgba(15,118,110,.18);
    }}
    .secondary {{ background: linear-gradient(135deg, var(--accent2), #9a3412); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(15,23,42,.05);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      margin: 0;
      overflow-x: auto;
    }}
    .cards {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .status-grid {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,.7);
    }}
    .metric strong {{ display: block; font-size: 1.15rem; margin-bottom: 4px; }}
    .story-grid {{ display: grid; gap: 16px; }}
    .story-card {{
      display: grid;
      gap: 0;
      grid-template-columns: 420px minmax(0, 1fr);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: rgba(255,255,255,.86);
    }}
    .story-media {{ position: relative; min-height: 320px; background: rgba(15,23,42,.04); }}
    .story-media img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .media-label {{
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 1;
      background: rgba(15,23,42,.72);
      color: white;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
    }}
    .story-body {{ padding: 16px; display: grid; gap: 12px; }}
    .story-head {{ display: flex; justify-content: space-between; align-items: start; gap: 12px; }}
    .story-head h3 {{ margin: 0; }}
    .story-head p {{ margin: 4px 0 0; }}
    .pill.small {{ padding: 6px 10px; font-size: 0.8rem; }}
    .asset-stack {{ display: grid; gap: 8px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .asset-chip {{
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(248,250,252,.86);
    }}
    .asset-chip img {{
      width: 76px;
      height: 76px;
      object-fit: cover;
      border-radius: 12px;
      background: rgba(15,23,42,.06);
    }}
    .preview-trigger {{
      border: 0;
      background: transparent;
      padding: 0;
      box-shadow: none;
      width: 100%;
    }}
    .preview-trigger img {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .preview-full {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .asset-chip span {{ display: block; color: var(--muted); font-size: 0.82rem; }}
    .asset-chip.empty {{ background: rgba(15,23,42,.03); }}
    .dialogue-list {{ margin: 0; padding-left: 18px; display: grid; gap: 12px; }}
    .dialogue-list li span {{ color: var(--muted); }}
    .shot-action {{ margin: 0; color: var(--muted); }}
    .dialogue-line {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 4px; }}
    .dialogue-line strong {{ margin-right: 8px; }}
    .audio-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 800;
      margin-left: 6px;
      background: rgba(15,23,42,.08);
      color: var(--ink);
    }}
    .audio-badge.available {{ background: rgba(22,163,74,.12); color: #166534; }}
    .audio-badge.missing {{ background: rgba(220,38,38,.12); color: #b91c1c; }}
    .audio-badge.duration {{ background: rgba(37,99,235,.12); color: #1d4ed8; }}
    .waveform {{
      display: flex;
      align-items: end;
      gap: 2px;
      min-width: 90px;
      height: 28px;
      padding: 2px 0;
    }}
    .waveform span {{
      display: block;
      width: 5px;
      border-radius: 999px;
      background: linear-gradient(180deg, #0f766e, #22c55e);
      opacity: 0.86;
    }}
    .missing {{
      border-left: 4px solid #dc2626;
      padding: 10px 12px;
      background: rgba(220,38,38,.06);
      border-radius: 12px;
    }}
    .preview-note {{ color: var(--muted); margin-top: 4px; }}
    .muted {{ color: var(--muted); }}
    .preview-modal {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }}
    .preview-modal.open {{ display: flex; }}
    .preview-backdrop {{
      position: absolute;
      inset: 0;
      background: rgba(15,23,42,.78);
      backdrop-filter: blur(4px);
    }}
    .preview-dialog {{
      position: relative;
      z-index: 1;
      max-width: min(92vw, 1100px);
      max-height: 90vh;
      background: white;
      border-radius: 20px;
      box-shadow: 0 30px 90px rgba(0,0,0,.35);
      padding: 16px;
      display: grid;
      gap: 10px;
    }}
    .preview-dialog img {{
      max-width: 100%;
      max-height: 78vh;
      border-radius: 16px;
      object-fit: contain;
      background: #f8fafc;
    }}
    .preview-title {{ font-weight: 800; color: var(--ink); }}
    .preview-close {{
      justify-self: end;
      border: 0;
      background: #0f766e;
      color: white;
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 800;
    }}
    @media (max-width: 900px) {{
      .hero, .forms, .grid, .cards {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="panel">
        <div class="pill">2DVideo Local Test UI</div>
        <h1>Pipeline control panel</h1>
        <p>Run the pipeline, import characters, and inspect outputs directly in the browser.</p>
        <p class="muted">{esc(message)}</p>
        <p><a href="/stories" style="color:#0f766e; font-weight:800; text-decoration:none;">Open story gallery</a> · <a href="/characters" style="color:#0f766e; font-weight:800; text-decoration:none;">Character library</a> · <a href="/prompts" style="color:#0f766e; font-weight:800; text-decoration:none;">Prompt generator</a></p>
      </div>
      <div class="panel status">
        <h2>Job Status</h2>
        <div class="progress">
          <div class="progress-bar"><div></div></div>
          <div><strong>{progress_pct:.1f}%</strong> {esc(progress_stage)} {esc(progress_message)}</div>
        </div>
        <pre>{esc(jobs_block)}</pre>
        <div class="status-grid">
          <button onclick="location.reload()" type="button">Refresh Status</button>
          <form method="post" action="/action" style="margin:0">
            <input type="hidden" name="action" value="stop_job" />
            <button type="submit" class="secondary">Stop Running Job</button>
          </form>
        </div>
        <div><strong>Registry characters:</strong> {len(registry)}</div>
        <div><strong>Voice entries:</strong> {len(voice_registry)}</div>
      </div>
    </div>

    <div class="panel form-card">
      <h2>Run Pipeline</h2>
      <div class="forms">
        <form method="post" action="/action">
          <input type="hidden" name="action" value="run_orchestrator" />
          <label>Episodes
            <input name="episodes" type="number" min="1" max="10" value="1" />
          </label>
          <label>Kind
            <select name="kind">
              <option value="poem">Poem</option>
              <option value="story">Story</option>
              <option value="both">Both</option>
            </select>
          </label>
          <label>Bootstrap demo cast
            <select name="bootstrap_demo">
              <option value="yes">Yes</option>
              <option value="no">No</option>
            </select>
          </label>
          <button type="submit">Run Pipeline</button>
        </form>

        <form method="post" action="/action">
          <input type="hidden" name="action" value="run_stage" />
          <label>Stage
            <select name="stage">
              <option value="story">01 Story Engine</option>
              <option value="asset_check">02 Asset Check</option>
              <option value="tts">03 TTS</option>
              <option value="rig_render">04 Rig Render</option>
              <option value="broll">05 B-Roll</option>
              <option value="upscale">06 Upscale</option>
              <option value="assemble">07 Assemble</option>
            </select>
          </label>
          <label>Episode path
            <input name="episode_path" placeholder="output/2026-07-01/2026-07-01-poem-01/episode.json" />
          </label>
          <button type="submit" class="secondary">Run Stage</button>
        </form>
      </div>
    </div>

    <div class="panel wide form-card">
      <h2>Story Browser</h2>
      <form method="get" action="/" style="display:grid; gap:12px; grid-template-columns: minmax(0, 1fr) auto; align-items:end;">
        <label>Episode
          <select name="episode">
            {''.join(episode_options) if episode_options else '<option value="">No episodes found</option>'}
          </select>
        </label>
        <button type="submit">Open Episode</button>
      </form>
      <p class="muted">Selected: {esc(selected_label)}</p>
      {f'<div class="status-grid" style="margin-bottom: 12px;"><div class="metric"><strong>{esc(selected_story_status["final_status"])}</strong><span>final output</span></div><div class="metric"><strong>{esc(selected_story_status["audio_mix_status"])}</strong><span>audio mix</span></div><div class="metric"><strong>{esc(selected_story_status["audio_missing_count"])}</strong><span>missing audio</span></div><div class="metric"><strong>{esc(selected_story_status["tts_count"])}</strong><span>TTS lines</span></div></div>' if selected_story_status else '<p class="muted">No selected episode available.</p>'}
      <div class="story-grid">
        {selected_storyboard if selected_storyboard else '<p class="muted">No storyboard available for this episode.</p>'}
      </div>
    </div>

    <div class="panel wide form-card">
      <h2>Compare Episodes</h2>
      <form method="get" action="/compare" style="display:grid; gap:12px; grid-template-columns: repeat(2, minmax(0, 1fr)) auto; align-items:end;">
        <label>Left
          <select name="left">
            {''.join(episode_options) if episode_options else '<option value="">No episodes found</option>'}
          </select>
        </label>
        <label>Right
          <select name="right">
            {''.join(episode_options) if episode_options else '<option value="">No episodes found</option>'}
          </select>
        </label>
        <button type="submit">Compare</button>
      </form>
    </div>

    <div class="grid">
      <div class="panel form-card">
        <h2>Import Character Sheet</h2>
        <form method="post" action="/action">
          <input type="hidden" name="action" value="import_character" />
          <label>Source image
            <input name="source" value="{esc((ROOT / 'config' / 'characters' / 'char_01_girl' / 'source_sheet.png'))}" />
          </label>
          <label>Character id
            <input name="character_id" value="char_01_girl" />
          </label>
          <label>Name
            <input name="name" value="Maya" />
          </label>
          <button type="submit">Import / Reimport</button>
        </form>
      </div>

      <div class="panel form-card">
        <h2>Latest Outputs</h2>
        <p><a href="/review">Open latest review page</a></p>
        {f'<div class="status-grid" style="margin-bottom: 12px;"><div class="metric"><strong>{esc(latest_story_status["final_status"])}</strong><span>final output</span></div><div class="metric"><strong>{esc(latest_story_status["audio_mix_status"])}</strong><span>audio mix</span></div><div class="metric"><strong>{esc(latest_story_status["audio_missing_count"])}</strong><span>missing audio</span></div><div class="metric"><strong>{esc(latest_story_status["tts_count"])}</strong><span>TTS lines</span></div></div>' if latest_story_status else '<p class="muted">No episode output yet.</p>'}
        <div class="cards">
          {''.join(latest_cards) if latest_cards else '<p class="muted">No output folders yet.</p>'}
        </div>
      </div>
    </div>

    {f'<div class="panel wide form-card"><h2>Latest Storyboard</h2><div class="story-grid">{latest_storyboard}</div></div>' if latest_storyboard else ''}

    <div class="grid">
      <div class="panel wide form-card">
        <h2>Character Registry</h2>
        <table>
          <thead><tr><th>ID</th><th>Name</th><th>Tier</th><th>Voice</th></tr></thead>
          <tbody>{''.join(registry_rows) if registry_rows else '<tr><td colspan="4">No characters found.</td></tr>'}</tbody>
        </table>
      </div>

      <div class="panel wide form-card">
        <h2>Voice Registry</h2>
        <table>
          <thead><tr><th>ID</th><th>Reference Clip</th><th>Sample Clip</th><th>Language</th></tr></thead>
          <tbody>{''.join(voice_rows) if voice_rows else '<tr><td colspan="4">No voice entries found.</td></tr>'}</tbody>
        </table>
      </div>

      <div class="panel wide form-card">
        <h2>Latest Daily Summary</h2>
        <pre>{esc(latest_summary or 'No summary yet.')}</pre>
      </div>
    </div>
  </div>
  {_preview_modal_markup()}
</body>
</html>"""


def _render_stories_page() -> str:
    episode_dirs = _episode_dirs(limit=60)
    cards: list[str] = []
    for episode_dir in episode_dirs:
        episode = _read_json(episode_dir / "episode.json") or {}
        story_status = _story_status(episode_dir)
        shots = episode.get("shots", [])
        first_shot = shots[0] if shots else {}
        bg_name = str(first_shot.get("background", "meadow_day"))
        bg_path = _background_preview_path(bg_name)
        cover_title = _episode_label(episode_dir)
        cards.append(
            f"""
            <article class="gallery-card">
              <button type="button" class="preview-trigger preview-full gallery-cover" onclick="openPreview('{_safe_js_string(_rel_file_url(bg_path))}', '{_safe_js_string(cover_title)}')">
                <img src="{_rel_file_url(bg_path)}" alt="{html.escape(cover_title)} cover" />
              </button>
              <div class="gallery-body">
                <div class="gallery-head">
                  <div>
                    <h3>{html.escape(cover_title)}</h3>
                    <p>{html.escape(episode.get('type', 'episode'))} · {html.escape(str(len(shots)))} shots</p>
                  </div>
                  <div class="pill small">{html.escape(story_status['final_status'])}</div>
                </div>
                <p class="gallery-meta">Audio mix: {html.escape(story_status['audio_mix_status'])} · Missing audio: {html.escape(str(story_status['audio_missing_count']))}</p>
                <div class="gallery-links">
                  <a href="/?episode={quote(str(episode_dir.relative_to(ROOT)))}">Open browser</a>
                  <a href="/review?episode={quote(str(episode_dir.relative_to(ROOT)))}">Review</a>
                </div>
              </div>
            </article>
            """
        )
    if not cards:
        cards_html = '<p class="muted">No episode folders found yet.</p>'
    else:
        cards_html = "".join(cards)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stories</title>
  <style>
    :root {{
      --ink: #1e293b;
      --muted: #5b6472;
      --line: rgba(30,41,59,.12);
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(135deg, #eef7f5, #f8efe0);
    }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 28px 18px 60px; }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 20px;
    }}
    .topbar a {{
      text-decoration: none;
      color: white;
      background: #0f766e;
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 800;
    }}
    h1 {{ margin: 0; font-size: 2rem; }}
    p {{ color: var(--muted); }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .gallery-card {{
      display: grid;
      grid-template-columns: 1.05fr 1fr;
      background: rgba(255,255,255,.86);
      border: 1px solid var(--line);
      border-radius: 20px;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(15,23,42,.12);
    }}
    .gallery-cover {{
      border: 0;
      padding: 0;
      background: transparent;
    }}
    .gallery-cover img {{
      width: 100%;
      height: 100%;
      min-height: 240px;
      object-fit: cover;
      display: block;
    }}
    .gallery-body {{
      padding: 18px;
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .gallery-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }}
    .gallery-head h3 {{ margin: 0; font-size: 1.15rem; }}
    .gallery-head p {{ margin: 4px 0 0; }}
    .gallery-meta {{ margin: 0; }}
    .gallery-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .gallery-links a {{
      text-decoration: none;
      color: white;
      background: #115e59;
      padding: 9px 12px;
      border-radius: 999px;
      font-weight: 800;
    }}
    .preview-modal {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }}
    .preview-modal.open {{ display: flex; }}
    .preview-backdrop {{
      position: absolute;
      inset: 0;
      background: rgba(15,23,42,.78);
      backdrop-filter: blur(4px);
    }}
    .preview-dialog {{
      position: relative;
      z-index: 1;
      max-width: min(92vw, 1100px);
      max-height: 90vh;
      background: white;
      border-radius: 20px;
      box-shadow: 0 30px 90px rgba(0,0,0,.35);
      padding: 16px;
      display: grid;
      gap: 10px;
    }}
    .preview-dialog img {{
      max-width: 100%;
      max-height: 78vh;
      border-radius: 16px;
      object-fit: contain;
      background: #f8fafc;
    }}
    .preview-title {{ font-weight: 800; color: var(--ink); }}
    .preview-close {{
      justify-self: end;
      border: 0;
      background: #0f766e;
      color: white;
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 800;
    }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 900px) {{
      .gallery-grid, .gallery-card {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>Stories</h1>
        <p>Browse every generated episode as a visual gallery.</p>
      </div>
      <div style="display:flex; gap:10px; flex-wrap:wrap;">
        <a href="/">Back to UI</a>
        <a href="/characters">Characters</a>
      </div>
    </div>
    <div class="gallery-grid">
      {cards_html}
    </div>
  </div>
  {_preview_modal_markup()}
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _serve_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        mime = mimetypes.types_map.get(suffix, "application/octet-stream")
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/":
            selected_episode = parse_qs(parsed.query).get("episode", [""])[0]
            page = _render_page(selected_episode_raw=selected_episode)
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return
        if path == "/review":
            episode = parse_qs(parsed.query).get("episode", [""])[0]
            episode_dir = _safe_path(episode) if episode else (_latest_episode_dir() or OUTPUT_DIR)
            page = _render_review_page(episode_dir)
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return
        if path == "/characters":
            page = _render_character_library_page()
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return
        if path == "/compare":
            query = parse_qs(parsed.query)
            left = _resolve_episode_dir(query.get("left", [""])[0]) or (_latest_episode_dir() or OUTPUT_DIR)
            right = _resolve_episode_dir(query.get("right", [""])[0]) or left
            page = _render_compare_page(left, right)
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return
        if path == "/stories":
            page = _render_stories_page()
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return
        if path == "/file":
            file_path = parse_qs(parsed.query).get("path", [""])[0]
            if not file_path:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = _safe_path(file_path)
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.types_map.get(path.suffix.lower(), "application/octet-stream"))
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:  # noqa: N802
        global FLASH_MESSAGE
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/":
            selected_episode = parse_qs(parsed.query).get("episode", [""])[0]
            page = _render_page(message=FLASH_MESSAGE if FLASH_MESSAGE else "", selected_episode_raw=selected_episode)
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/review":
            episode = parse_qs(parsed.query).get("episode", [""])[0]
            episode_dir = _safe_path(episode) if episode else (_latest_episode_dir() or OUTPUT_DIR)
            page = _render_review_page(episode_dir)
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/characters":
            page = _render_character_library_page()
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/prompts":
            page = _render_prompt_generator_page(parse_qs(parsed.query))
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/compare":
            query = parse_qs(parsed.query)
            left = _resolve_episode_dir(query.get("left", [""])[0]) or (_latest_episode_dir() or OUTPUT_DIR)
            right = _resolve_episode_dir(query.get("right", [""])[0]) or left
            page = _render_compare_page(left, right)
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/stories":
            page = _render_stories_page()
            data = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/file":
            file_path = parse_qs(parsed.query).get("path", [""])[0]
            if not file_path:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = _safe_path(file_path)
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(path)
            return
        message = FLASH_MESSAGE
        FLASH_MESSAGE = ""
        page = _render_page(message=message)
        data = page.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/action":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body)
        action = form.get("action", [""])[0]
        message = ""
        try:
            if action == "run_orchestrator":
                episodes = form.get("episodes", ["1"])[0]
                kind = form.get("kind", ["poem"])[0]
                bootstrap_demo = form.get("bootstrap_demo", ["yes"])[0] == "yes"
                command = [
                    sys.executable,
                    str(ROOT / "pipeline" / "orchestrator.py"),
                    "--episodes",
                    episodes,
                    "--kind",
                    kind,
                ]
                if bootstrap_demo:
                    command.append("--bootstrap-demo")
                start_job(command, f"orchestrator ({kind})")
                message = f"Started orchestrator for {episodes} episode(s)."
            elif action == "run_stage":
                stage = form.get("stage", ["story"])[0]
                episode_path = form.get("episode_path", [""])[0].strip()
                if not episode_path:
                    raise ValueError("episode_path is required for stage runs.")
                script = {
                    "story": "01_story_engine.py",
                    "asset_check": "02_asset_check.py",
                    "tts": "03_tts.py",
                    "rig_render": "04_rig_render.py",
                    "broll": "05_broll_gen.py",
                    "upscale": "06_upscale.py",
                    "assemble": "07_assemble.py",
                }[stage]
                command = [sys.executable, str(ROOT / "pipeline" / script)]
                if stage == "story":
                    command += ["--template", str(ROOT / "config" / "episode_templates" / "poem_template.json"), "--output", episode_path]
                elif stage == "upscale":
                    command += ["--input", episode_path, "--output", str(Path(episode_path).with_name("upscaled.mp4"))]
                elif stage == "assemble":
                    command += ["--episode", episode_path, "--output", str(Path(episode_path).with_name("assembled.mp4"))]
                else:
                    command += ["--episode", episode_path]
                start_job(command, f"stage: {stage}")
                message = f"Started stage {stage}."
            elif action == "import_character":
                source = form.get("source", [""])[0].strip()
                character_id = form.get("character_id", [""])[0].strip()
                name = form.get("name", [""])[0].strip()
                if not source or not character_id or not name:
                    raise ValueError("source, character_id, and name are required.")
                command = [
                    sys.executable,
                    str(ROOT / "pipeline" / "import_character_sheet.py"),
                    "--source",
                    source,
                    "--character-id",
                    character_id,
                    "--name",
                    name,
                ]
                start_job(command, f"import: {character_id}")
                message = f"Started import for {character_id}."
            elif action == "save_episode_meta":
                episode_path = form.get("episode_path", [""])[0].strip()
                story_status = form.get("story_status", ["draft"])[0].strip() or "draft"
                notes = form.get("notes", [""])[0]
                if not episode_path:
                    raise ValueError("episode_path is required.")
                episode_dir = _resolve_episode_dir(episode_path)
                if episode_dir is None:
                    raise ValueError("Could not resolve episode directory.")
                _write_episode_meta(episode_dir, status=story_status, notes=notes)
                message = f"Saved story notes for {episode_dir.name}."
            elif action == "stop_job":
                stop_job()
                message = "Stopped current job."
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as exc:
            message = f"Error: {exc}"
        global FLASH_MESSAGE
        FLASH_MESSAGE = message
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the 2DVideo browser UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"2DVideo UI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
