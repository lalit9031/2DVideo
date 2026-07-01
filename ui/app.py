from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import mimetypes
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlparse


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
        <h2>Preview Video</h2>
        {f'<video controls src="/file?path={quote(str(final_mp4.relative_to(ROOT)))}"></video>' if final_mp4.exists() else '<p>No final video available.</p>'}
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
</body>
</html>"""


def _read_progress() -> dict[str, Any]:
    try:
        return json.loads(UI_PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _render_page(message: str = "") -> str:
    with LOCK:
        state = STATE.as_dict()
    registry = _read_json(CHAR_DIR / "registry.json") or {}
    voice_registry = _read_json(VOICE_DIR / "voice_registry.json") or {}
    latest_dirs = _latest_output_dirs()
    latest_episode = _latest_episode_dir()
    latest_summary = None
    if latest_dirs:
        summary_path = latest_dirs[0] / "daily_summary.json"
        if summary_path.exists():
            latest_summary = _read_text(summary_path, 10000)
    progress = _read_progress()
    progress_pct = float(progress.get("percent", state.get("progress") or 0.0) or 0.0)
    progress_stage = progress.get("stage", state.get("stage") or "")
    progress_message = progress.get("message", state.get("message") or "")
    auto_refresh = state["status"] == "running" or progress.get("status") == "running"
    jobs_block = _job_summary(progress)

    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

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
            f"<tr><td>{esc(key)}</td><td>{esc(item.get('reference_clip'))}</td><td>{esc(item.get('language'))}</td></tr>"
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
    .muted {{ color: var(--muted); }}
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

    <div class="grid">
      <div class="panel form-card">
        <h2>Import Character Sheet</h2>
        <form method="post" action="/action">
          <input type="hidden" name="action" value="import_character" />
          <label>Source image
            <input name="source" value="{esc((ROOT / 'Char' / 'Gemini_Generated_Image_i0ao1ai0ao1ai0ao.png'))}" />
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
        <div class="cards">
          {''.join(latest_cards) if latest_cards else '<p class="muted">No output folders yet.</p>'}
        </div>
      </div>
    </div>

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
          <thead><tr><th>ID</th><th>Reference Clip</th><th>Language</th></tr></thead>
          <tbody>{''.join(voice_rows) if voice_rows else '<tr><td colspan="3">No voice entries found.</td></tr>'}</tbody>
        </table>
      </div>

      <div class="panel wide form-card">
        <h2>Latest Daily Summary</h2>
        <pre>{esc(latest_summary or 'No summary yet.')}</pre>
      </div>
    </div>
  </div>
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
            page = _render_page()
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
            page = _render_page(message=FLASH_MESSAGE if FLASH_MESSAGE else "")
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
