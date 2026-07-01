"""
ui/app.py — Fruit Cartoon Studio UI
-------------------------------------
Minimal single-page web UI with:
  • 2 run buttons: Short Video (9:16) | Full Length (16:9)
  • Terminal-style live progress log with per-stage status icons
  • Thumbnail grid: shows scene images as they're generated
  • Final preview player + Download button

Streams live progress via Server-Sent Events (SSE).
"""
from __future__ import annotations

import html
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pipeline.common import OUTPUT_DIR, write_json

ROOT = Path(__file__).resolve().parents[1]
UI_PROGRESS_FILE = OUTPUT_DIR / ".ui_progress.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# ── Global job state ──────────────────────────────────────────────────────────
_job_lock = threading.Lock()
_job: dict[str, Any] = {
    "status": "idle",
    "format": "",
    "percent": 0.0,
    "stage": "",
    "message": "",
    "log": [],          # list of {stage, status, message, preview}
    "previews": [],     # list of PNG paths generated so far
    "final_path": "",
    "error": "",
}
_job_proc: subprocess.Popen | None = None

_interactive_state: dict[str, Any] = {
    "active": False,
    "format": "full",
    "episode_id": "",
    "current_step": 0,
    "status": "idle",
    "message": "",
    "error": "",
    "episode_dir": "",
    "shots": []
}


# ── Stage display metadata ────────────────────────────────────────────────────
STAGE_LABELS = {
    "bootstrap":    "🚀 Pipeline initialising",
    "story":        "📝 Story & Prompts (Gemma 3 12B)",
    "asset_check":  "✅ Asset Check",
    "tts":          "🎙️ Hindi Voice Synthesis (XTTS-v2)",
    "scene_gen":    "🎨 Scene Image Generation (sd-server-vulkan · 22 steps)",
    "upscale":      "🔍 Upscaling 2× (Real-ESRGAN)",
    "animatediff":  "🎬 Animating Scenes (AnimateDiff-Lightning)",
    "broll":        "🌿 B-roll Generation (LTX-2.3)",
    "assemble":     "🎞️ Final Assembly (ffmpeg)",
    "qa":           "🧪 QA Check",
    "complete":     "🎉 Complete!",
    "failed":       "❌ Failed",
}


# ── HTML page ─────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🍎 Fruit Cartoon Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
    --border: #30363d; --border2: #444c56;
    --text: #e6edf3; --text2: #8b949e; --text3: #6e7681;
    --green: #3fb950; --blue: #58a6ff; --orange: #f78166;
    --yellow: #e3b341; --purple: #bc8cff; --pink: #ff7b72;
    --accent: #388bfd;
    --short-color: #ff6b6b; --full-color: #4ecdc4;
    --radius: 12px; --radius-sm: 8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'Inter', sans-serif; font-size: 14px;
    min-height: 100vh; display: flex; flex-direction: column;
  }

  /* ── Header ── */
  .header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex; align-items: center; gap: 16px;
  }
  .header-icon { font-size: 36px; }
  .header-title { font-size: 22px; font-weight: 700; }
  .header-sub { font-size: 12px; color: var(--text2); margin-top: 2px; }

  /* ── Tabs ── */
  .tabs {
    display: flex; gap: 8px; border-bottom: 1px solid var(--border);
    margin-bottom: 24px; padding: 0 32px; background: var(--bg2);
  }
  .tab-btn {
    background: transparent; border: none; color: var(--text2);
    padding: 16px 24px; cursor: pointer; font-size: 14px; font-weight: 600;
    transition: all 0.2s; border-bottom: 2px solid transparent; margin-bottom: -1px;
  }
  .tab-btn.active {
    color: var(--blue); border-bottom: 2px solid var(--blue);
  }
  .tab-btn:hover:not(.active) {
    color: var(--text);
  }

  /* ── Main layout ── */
  .main { flex: 1; padding: 28px 32px; max-width: 1100px; margin: 0 auto; width: 100%; }

  /* Tab Sections */
  .tab-content { display: none; }
  .tab-content.visible { display: block; }

  /* ── Format selector ── */
  .format-section { margin-bottom: 32px; }
  .format-section h2 { font-size: 13px; font-weight: 600; color: var(--text2); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 16px; }
  .format-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .format-btn {
    border: 2px solid var(--border2); border-radius: var(--radius);
    padding: 24px; cursor: pointer; background: var(--bg2);
    transition: all 0.2s; display: flex; flex-direction: column; align-items: center; gap: 8px;
    position: relative; overflow: hidden; color: var(--text); width: 100%;
  }
  .format-btn::before {
    content: ''; position: absolute; inset: 0; opacity: 0;
    transition: opacity 0.2s; z-index: 1;
  }
  .format-btn.short:hover { border-color: var(--short-color); box-shadow: 0 0 20px rgba(255,107,107,0.15); }
  .format-btn.full:hover { border-color: var(--full-color); box-shadow: 0 0 20px rgba(78,205,196,0.15); }
  .format-btn.short.selected { border-color: var(--short-color); background: rgba(255,107,107,0.05); }
  .format-btn.full.selected { border-color: var(--full-color); background: rgba(78,205,196,0.05); }
  .btn-icon { font-size: 40px; z-index: 2; }
  .btn-title { font-size: 18px; font-weight: 700; z-index: 2; }
  .btn-badge {
    font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 20px;
    background: var(--bg3); border: 1px solid var(--border); z-index: 2;
  }
  .format-btn.short .btn-badge { color: var(--short-color); }
  .format-btn.full .btn-badge { color: var(--full-color); }
  .btn-sub { font-size: 12px; color: var(--text2); line-height: 1.5; z-index: 2; margin-top: 8px; text-align: center; }

  /* ── Progress bar ── */
  .progress-section {
    background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 20px 24px; margin-bottom: 24px; display: none;
  }
  .progress-section.visible { display: block; }
  .progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .progress-label { font-weight: 600; color: var(--text); }
  .progress-pct { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--blue); }
  .progress-bar-bg { background: var(--bg3); height: 8px; border-radius: 4px; overflow: hidden; }
  .progress-bar-fill { background: linear-gradient(90deg, var(--blue) 0%, var(--purple) 100%); height: 100%; transition: width 0.3s; }

  /* ── Stage log ── */
  .stage-log {
    background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
    margin-bottom: 24px; display: none; overflow: hidden;
  }
  .stage-log.visible { display: block; }
  .log-header { padding: 14px 20px; background: var(--bg3); border-bottom: 1px solid var(--border); font-size: 13px; font-weight: 700; }
  .log-entries { padding: 16px 20px; max-height: 240px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }
  .log-entry {
    display: flex; gap: 12px; padding: 10px 12px; border-radius: var(--radius-sm);
    background: var(--bg3); border: 1px solid transparent; font-size: 13px; transition: all 0.2s;
  }
  .log-entry.active { border-color: var(--blue); background: rgba(88,166,255,0.05); }
  .log-entry.done { border-color: var(--border); background: var(--bg2); opacity: 0.85; }
  .log-entry.failed { border-color: var(--pink); background: rgba(255,123,114,0.05); }
  .log-icon { font-size: 16px; margin-top: 1px; }
  .log-content { flex: 1; display: flex; flex-direction: column; gap: 4px; }
  .log-stage { font-weight: 600; color: var(--text); }
  .log-msg { color: var(--text2); font-size: 12px; }
  .log-prompt {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text3);
    padding: 6px 10px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; margin-top: 6px; word-break: break-all;
  }

  /* ── Scene previews ── */
  .previews-section { margin-bottom: 24px; display: none; }
  .previews-section.visible { display: block; }
  .previews-header {
    font-size: 13px; font-weight: 600; color: var(--text2);
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px;
  }
  .previews-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px;
  }
  .preview-card {
    border-radius: var(--radius-sm); overflow: hidden; aspect-ratio: 16/9;
    background: var(--bg3); border: 1px solid var(--border);
    position: relative; cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .preview-card.portrait { aspect-ratio: 9/16; }
  .preview-card:hover { transform: scale(1.04); box-shadow: 0 6px 20px rgba(0,0,0,0.5); }
  .preview-card img { width: 100%; height: 100%; object-fit: cover; }
  .preview-card .shot-label {
    position: absolute; bottom: 4px; left: 4px;
    background: rgba(0,0,0,0.7); color: #fff; font-size: 10px;
    padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono';
  }
  .preview-card .pending {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    color: var(--text3); font-size: 22px;
  }

  /* ── Final output ── */
  .final-section { margin-top: 24px; display: none; }
  .final-section.visible { display: block; }
  .final-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
  }
  .final-header {
    padding: 14px 20px; background: var(--bg3);
    border-bottom: 1px solid var(--border);
    font-size: 14px; font-weight: 700;
  }
  .final-body { padding: 20px; }
  video { width: 100%; border-radius: var(--radius-sm); background: #000; max-height: 360px; }
  .final-actions { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
  .action-btn {
    padding: 10px 20px; border-radius: var(--radius-sm);
    font-size: 13px; font-weight: 600; cursor: pointer;
    border: none; transition: all 0.15s; display: flex; align-items: center; gap: 6px;
    text-decoration: none;
  }
  .action-btn.primary   { background: var(--green); color: #000; }
  .action-btn.secondary { background: var(--bg3); color: var(--text); border: 1px solid var(--border2); }
  .action-btn.danger { background: rgba(255,123,114,0.15); color: var(--pink); border: 1px solid var(--pink); }
  .action-btn:hover { filter: brightness(1.15); transform: translateY(-1px); }

  /* ── Stop button ── */
  .stop-btn {
    padding: 8px 18px; border-radius: var(--radius-sm); border: 1px solid var(--pink);
    background: transparent; color: var(--pink); font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all 0.15s; display: none;
  }
  .stop-btn.visible { display: inline-flex; align-items: center; gap: 6px; }
  .stop-btn:hover { background: rgba(255,123,114,0.1); }

  /* ── Interactive studio layout ── */
  .setup-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 24px; margin-bottom: 24px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .setup-row { display: flex; gap: 16px; align-items: center; }
  .setup-select {
    background: var(--bg3); border: 1px solid var(--border); color: var(--text);
    padding: 10px 16px; border-radius: var(--radius-sm); font-size: 14px; font-weight: 600;
  }
  .studio-status-banner {
    padding: 12px 20px; border-radius: var(--radius-sm); font-weight: 600; font-size: 13px;
    margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;
  }
  .studio-status-banner.running { background: rgba(88,166,255,0.12); color: var(--blue); border: 1px solid var(--blue); }
  .studio-status-banner.failed { background: rgba(255,123,114,0.12); color: var(--pink); border: 1px solid var(--pink); }
  .studio-status-banner.idle { background: var(--bg2); color: var(--text2); border: 1px solid var(--border); }
  
  .timeline { display: flex; flex-direction: column; gap: 16px; margin-top: 12px; }
  .timeline-step {
    border: 1px solid var(--border); border-radius: var(--radius);
    background: var(--bg2); overflow: hidden; opacity: 0.5; pointer-events: none;
    transition: all 0.2s;
  }
  .timeline-step.active {
    opacity: 1; pointer-events: auto; border: 1.5px solid var(--blue);
    box-shadow: 0 4px 15px rgba(88,166,255,0.06);
  }
  .timeline-step.completed {
    opacity: 1; pointer-events: auto; border: 1px solid var(--green);
  }
  .step-header {
    padding: 16px 20px; background: var(--bg3); display: flex; align-items: center;
    justify-content: space-between; cursor: pointer; font-weight: 700;
  }
  .step-title { display: flex; align-items: center; gap: 10px; font-size: 15px; }
  .step-status-icon { font-size: 16px; }
  .step-body { padding: 20px; display: none; border-top: 1px solid var(--border); }
  .timeline-step.active .step-body { display: block; }
  .timeline-step.completed .step-body { display: block; }

  /* Shot grid editor */
  .shot-editor-card {
    background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 16px; margin-bottom: 12px; display: flex; flex-direction: column; gap: 12px;
  }
  .shot-editor-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 768px) {
    .shot-editor-grid { grid-template-columns: 3fr 2fr; gap: 16px; }
  }
  .editor-fields-col { display: flex; flex-direction: column; gap: 12px; }
  .editor-preview-col { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; border-left: 1px solid var(--border); padding-left: 16px; }
  .editor-field { display: flex; flex-direction: column; gap: 6px; }
  .editor-field label { font-size: 11px; text-transform: uppercase; color: var(--text2); font-weight: 700; letter-spacing: 0.6px; }
  .editor-input {
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 6px; font-size: 13px; font-family: inherit; width: 100%;
  }
  .editor-textarea {
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 6px; font-size: 13px; font-family: inherit; width: 100%;
    resize: vertical; min-height: 50px;
  }
  .shot-preview-img { width: 100%; max-height: 180px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); }
  .shot-preview-img.portrait { max-height: 240px; aspect-ratio: 9/16; }
  .shot-preview-video { width: 100%; max-height: 180px; background: #000; border-radius: 6px; }
  .shot-preview-video.portrait { max-height: 240px; aspect-ratio: 9/16; }
  
  .step-actions { display: flex; gap: 12px; margin-top: 16px; justify-content: flex-end; }

  /* ── Footer ── */
  .footer {
    padding: 12px 32px; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    font-size: 11px; color: var(--text3);
  }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block;
    margin-right: 6px; background: var(--text3);
  }
  .status-dot.running { background: var(--blue); animation: pulse 1.2s ease-in-out infinite; }
  .status-dot.done    { background: var(--green); }
  .status-dot.failed  { background: var(--pink); }

  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-icon">🍎</div>
  <div>
    <div class="header-title">Fruit Cartoon Studio</div>
    <div class="header-sub">Hindi AI Animation · Gemma 3 12B · DreamShaperXL · AnimateDiff-Lightning</div>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" id="tabBtnAuto" onclick="switchTab('auto')">🤖 Full Automation</button>
  <button class="tab-btn" id="tabBtnStudio" onclick="switchTab('studio')">🎬 Interactive Studio (Step-by-Step)</button>
</div>

<div class="main">

  <!-- ==================== TAB 1: AUTO PIPELINE ==================== -->
  <div id="autoTab" class="tab-content visible">
    <!-- Format selector -->
    <div class="format-section" id="formatSection">
      <h2>Select Video Format</h2>
      <div class="format-buttons">
        <button class="format-btn short" id="btnShort" onclick="runPipeline('short')">
          <div class="btn-icon">📱</div>
          <div class="btn-title">Short Video</div>
          <div class="btn-badge">9:16 · YouTube Shorts</div>
          <div class="btn-sub">60–90 sec poem<br>Portrait orientation<br>2K upscaled (1152×2048)</div>
        </button>
        <button class="format-btn full" id="btnFull" onclick="runPipeline('full')">
          <div class="btn-icon">🖥️</div>
          <div class="btn-title">Full Length Video</div>
          <div class="btn-badge">16:9 · YouTube Main</div>
          <div class="btn-sub">5–10 min story<br>Landscape orientation<br>2K upscaled (2048×1152)</div>
        </button>
      </div>
      <div style="text-align:right;margin-top:12px">
        <button class="stop-btn" id="stopBtn" onclick="stopPipeline()">⏹ Stop</button>
      </div>
    </div>

    <!-- Progress bar -->
    <div class="progress-section" id="progressSection">
      <div class="progress-header">
        <span class="progress-label" id="progressLabel">Running…</span>
        <span class="progress-pct" id="progressPct">0%</span>
      </div>
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressFill" style="width:0%"></div></div>
    </div>

    <!-- Stage log -->
    <div class="stage-log" id="stageLog">
      <div class="log-header">📋 Pipeline Log</div>
      <div class="log-entries" id="logEntries"></div>
    </div>

    <!-- Scene previews -->
    <div class="previews-section" id="previewsSection">
      <div class="previews-header">🖼️ Generated Scene Images</div>
      <div class="previews-grid" id="previewsGrid"></div>
    </div>

    <!-- Final output -->
    <div class="final-section" id="finalSection">
      <div class="final-card">
        <div class="final-header">🎉 Episode Ready!</div>
        <div class="final-body">
          <video id="finalVideo" controls></video>
          <div class="final-actions">
            <a id="downloadBtn" class="action-btn primary" href="#" download>⬇ Download MP4</a>
            <button class="action-btn secondary" onclick="resetUI()">🔄 Run New Episode</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ==================== TAB 2: INTERACTIVE STUDIO ==================== -->
  <div id="studioTab" class="tab-content">
    <!-- Project start banner -->
    <div id="studioSetup" class="setup-card">
      <h2 style="font-size:16px;font-weight:700">Start Step-by-Step Project</h2>
      <p style="color:var(--text2);font-size:13px;line-height:1.5">
        Run and verify each AI stage individually. Preview storyboards, play Hindi audio lines, edit Stable Diffusion prompts, regenerate individual scenes, and animate shot clips with GPU control.
      </p>
      <div class="setup-row">
        <span style="font-weight:600">Choose Orientation:</span>
        <select id="studioFormat" class="setup-select">
          <option value="full">Landscape (16:9 - Main Video)</option>
          <option value="short">Portrait (9:16 - Shorts)</option>
        </select>
        <button class="action-btn primary" onclick="studioStart()">🚀 Start New Project</button>
      </div>
    </div>

    <!-- Interactive Workspace -->
    <div id="studioWorkspace" style="display:none">
      <div class="studio-status-banner idle" id="studioStatusBanner">
        <span id="studioBannerMsg">Workspace Loaded</span>
        <span id="studioBannerProgress" style="font-family:'JetBrains Mono';font-size:11px"></span>
      </div>

      <div class="timeline">
        <!-- Step 1: Ollama Script -->
        <div class="timeline-step" id="stepCard1">
          <div class="step-header" onclick="toggleStepBody(1)">
            <span class="step-title"><span class="step-num-badge">1️⃣</span> Step 1: Script & Storyboard Prompting (Ollama)</span>
            <span class="step-status-icon" id="stepIcon1">🔒</span>
          </div>
          <div class="step-body" id="stepBody1">
            <p style="color:var(--text2);margin-bottom:16px;line-height:1.5">Generates the characters, dialogue list, and Stable Diffusion scene prompts using Gemma 3 12B via local Ollama.</p>
            <div id="stepData1"></div>
            <div class="step-actions">
              <button class="action-btn secondary" onclick="studioRunStep(1)">🔄 Generate / Re-run Script</button>
              <button class="action-btn primary" id="stepNextBtn1" onclick="studioSaveAndContinue(1)">✓ Approve Script & Continue</button>
            </div>
          </div>
        </div>

        <!-- Step 2: Hindi Voice (TTS) -->
        <div class="timeline-step" id="stepCard2">
          <div class="step-header" onclick="toggleStepBody(2)">
            <span class="step-title"><span class="step-num-badge">2️⃣</span> Step 2: Voice Over Synthesis (XTTS-v2)</span>
            <span class="step-status-icon" id="stepIcon2">🔒</span>
          </div>
          <div class="step-body" id="stepBody2">
            <p style="color:var(--text2);margin-bottom:16px;line-height:1.5">Synthesizes high-fidelity Hindi speech audio clips for each character dialogue line.</p>
            <div id="stepData2"></div>
            <div class="step-actions">
              <button class="action-btn secondary" onclick="studioRunStep(2)">🔄 Synthesize Voices</button>
              <button class="action-btn primary" id="stepNextBtn2" onclick="studioSaveAndContinue(2)">✓ Approve Voices & Continue</button>
            </div>
          </div>
        </div>

        <!-- Step 3: Scene Generation (SD) -->
        <div class="timeline-step" id="stepCard3">
          <div class="step-header" onclick="toggleStepBody(3)">
            <span class="step-title"><span class="step-num-badge">3️⃣</span> Step 3: Scene Image Generation (Stable Diffusion)</span>
            <span class="step-status-icon" id="stepIcon3">🔒</span>
          </div>
          <div class="step-body" id="stepBody3">
            <p style="color:var(--text2);margin-bottom:16px;line-height:1.5">Generates a Stable Diffusion scene illustration for each shot (22 steps, DreamShaperXL). Tweak prompts and regenerate individual images as needed.</p>
            <div id="stepData3"></div>
            <div class="step-actions">
              <button class="action-btn secondary" onclick="studioRunStep(3)">🔄 Generate Scene Images</button>
              <button class="action-btn primary" id="stepNextBtn3" onclick="studioSaveAndContinue(3)">✓ Approve Images & Continue</button>
            </div>
          </div>
        </div>

        <!-- Step 4: 2K Upscaling -->
        <div class="timeline-step" id="stepCard4">
          <div class="step-header" onclick="toggleStepBody(4)">
            <span class="step-title"><span class="step-num-badge">4️⃣</span> Step 4: 2K Upscaling (OpenCV Lanczos)</span>
            <span class="step-status-icon" id="stepIcon4">🔒</span>
          </div>
          <div class="step-body" id="stepBody4">
            <p style="color:var(--text2);margin-bottom:16px;line-height:1.5">Upscales the scene illustrations to 2K resolution in landscape (2048x1152) or portrait (1152x2048) dimensions.</p>
            <div id="stepData4"></div>
            <div class="step-actions">
              <button class="action-btn secondary" onclick="studioRunStep(4)">🔍 Upscale Images</button>
              <button class="action-btn primary" id="stepNextBtn4" onclick="studioSaveAndContinue(4)">✓ Approve & Continue to Animation</button>
            </div>
          </div>
        </div>

        <!-- Step 5: Animation (AnimateDiff) -->
        <div class="timeline-step" id="stepCard5">
          <div class="step-header" onclick="toggleStepBody(5)">
            <span class="step-title"><span class="step-num-badge">5️⃣</span> Step 5: Video Animation (AnimateDiff-Lightning)</span>
            <span class="step-status-icon" id="stepIcon5">🔒</span>
          </div>
          <div class="step-body" id="stepBody5">
            <p style="color:var(--text2);margin-bottom:16px;line-height:1.5">Applies motion to your upscaled images using GPU image-to-video (strength 0.75, 4 steps). Play clips and regenerate individually if needed.</p>
            <div id="stepData5"></div>
            <div class="step-actions">
              <button class="action-btn secondary" onclick="studioRunStep(5)">🎬 Animate Shots</button>
              <button class="action-btn primary" id="stepNextBtn5" onclick="studioSaveAndContinue(5)">✓ Approve Animations & Continue</button>
            </div>
          </div>
        </div>

        <!-- Step 6: Final Assembly -->
        <div class="timeline-step" id="stepCard6">
          <div class="step-header" onclick="toggleStepBody(6)">
            <span class="step-title"><span class="step-num-badge">6️⃣</span> Step 6: Final Assembly & Rendering (ffmpeg)</span>
            <span class="step-status-icon" id="stepIcon6">🔒</span>
          </div>
          <div class="step-body" id="stepBody6">
            <p style="color:var(--text2);margin-bottom:16px;line-height:1.5">Assembles animated clips with voice audio tracks, normalizes sample rates, loops background music (if present), and burns subtitles.</p>
            <div id="stepData6"></div>
            <div class="step-actions" style="margin-top:20px">
              <button class="action-btn primary" onclick="studioRunStep(6)">🎞️ Assemble Final Video</button>
            </div>
          </div>
        </div>
      </div>

      <div style="margin-top:24px;display:flex;justify-content:flex-end">
        <button class="action-btn danger" onclick="studioReset()">❌ Reset & Close Workspace</button>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  <div>
    <span class="status-dot" id="statusDot"></span>
    <span id="statusText">Ready</span>
  </div>
  <div>AMD RX 7900 XTX · ROCm · Vulkan · Ollama</div>
</div>

<script>
const STAGE_LABELS = {
  bootstrap:   '🚀 Pipeline initialising',
  story:       '📝 Story & Prompts (Gemma 3 12B)',
  asset_check: '✅ Asset Check',
  tts:         '🎙️ Hindi Voice Synthesis',
  scene_gen:   '🎨 Scene Image Generation (22 steps)',
  upscale:     '🔍 Upscaling 2× (Real-ESRGAN)',
  animatediff: '🎬 AnimateDiff-Lightning',
  broll:       '🌿 B-roll (LTX-2.3)',
  assemble:    '🎞️ Final Assembly',
  qa:          '🧪 QA Check',
  complete:    '🎉 Complete!',
  failed:      '❌ Failed',
};

let es = null;
let currentFormat = '';
let stageMap = {};       // stage → log entry element
let previewsSeen = new Set();

// ── Tab Management ──────────────────────────────────────────────────────────
function switchTab(mode) {
  document.getElementById('tabBtnAuto').classList.toggle('active', mode === 'auto');
  document.getElementById('tabBtnStudio').classList.toggle('active', mode === 'studio');
  
  document.getElementById('autoTab').classList.toggle('visible', mode === 'auto');
  document.getElementById('studioTab').classList.toggle('visible', mode === 'studio');
  
  if (mode === 'studio') {
    studioFetchState();
  }
}

// ── Auto Pipeline JS ────────────────────────────────────────────────────────
function runPipeline(fmt) {
  currentFormat = fmt;
  document.getElementById('btnShort').disabled = true;
  document.getElementById('btnFull').disabled = true;
  document.getElementById('stopBtn').classList.add('visible');
  document.getElementById('progressSection').classList.add('visible');
  document.getElementById('stageLog').classList.add('visible');
  document.getElementById('finalSection').classList.remove('visible');
  document.getElementById('logEntries').innerHTML = '';
  document.getElementById('previewsGrid').innerHTML = '';
  stageMap = {};
  previewsSeen = new Set();

  setStatus('running', `Running ${fmt === 'short' ? 'Short 9:16' : 'Full Length 16:9'} pipeline…`);

  fetch('/api/run', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({format: fmt})
  }).catch(e => console.error('run failed', e));

  if (es) es.close();
  es = new EventSource('/api/progress-stream');
  es.onmessage = (e) => handleProgress(JSON.parse(e.data));
  es.onerror = () => { console.warn('SSE error'); };
}

function stopPipeline() {
  fetch('/api/stop', {method:'POST'}).catch(()=>{});
  if (es) { es.close(); es = null; }
  setStatus('idle', 'Stopped');
  resetButtons();
}

function resetButtons() {
  document.getElementById('btnShort').disabled = false;
  document.getElementById('btnFull').disabled = false;
  document.getElementById('stopBtn').classList.remove('visible');
}

function resetUI() {
  resetButtons();
  document.getElementById('progressSection').classList.remove('visible');
  document.getElementById('stageLog').classList.remove('visible');
  document.getElementById('previewsSection').classList.remove('visible');
  document.getElementById('finalSection').classList.remove('visible');
  setStatus('idle', 'Ready');
}

function setStatus(s, txt) {
  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot ' + s;
  document.getElementById('statusText').textContent = txt;
}

function handleProgress(data) {
  const pct = Math.round(data.percent || 0);
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressPct').textContent = pct + '%';
  document.getElementById('progressLabel').textContent =
    STAGE_LABELS[data.stage] || data.stage || 'Running…';

  // Update stage log entry
  upsertLogEntry(data);

  // Check for new scene preview images
  if (data.preview_image) addPreview(data.preview_image, data.shot_id || '');

  // Scan previews dir periodically via side-channel (data.previews list)
  if (data.previews && Array.isArray(data.previews)) {
    data.previews.forEach(p => addPreview(p.path, p.shot_id));
  }

  if (data.status === 'done') {
    if (es) { es.close(); es = null; }
    setStatus('done', 'Complete!');
    resetButtons();
    if (data.final_path) showFinal(data.final_path);
  } else if (data.status === 'failed') {
    if (es) { es.close(); es = null; }
    setStatus('failed', 'Failed — see log');
    resetButtons();
    upsertLogEntry({...data, stage: 'failed'});
  }
}

function upsertLogEntry(data) {
  const stage = data.stage || 'unknown';
  const status = data.status || 'running';

  let entry = stageMap[stage];
  if (!entry) {
    entry = document.createElement('div');
    entry.className = 'log-entry';
    stageMap[stage] = entry;
    document.getElementById('logEntries').appendChild(entry);
    document.getElementById('stageLog').classList.add('visible');
  }

  // Mark previous active entry as done
  Object.values(stageMap).forEach(e => {
    if (e !== entry && e.classList.contains('active')) e.classList.replace('active','done');
  });

  const icon = status === 'done' ? '✅' : status === 'failed' ? '❌' : stage === 'complete' ? '🎉' : '🔄';
  const cls  = status === 'failed' ? 'failed' : (status === 'done' || stage === 'complete') ? 'done' : 'active';
  entry.className = 'log-entry ' + cls;

  const promptSnip = data.message && data.message.length > 20 && data.stage === 'scene_gen'
    ? `<div class="log-prompt">${escHtml(data.message)}</div>` : '';

  entry.innerHTML = `
    <div class="log-icon">${icon}</div>
    <div class="log-content">
      <div class="log-stage">${escHtml(STAGE_LABELS[stage] || stage)}</div>
      <div class="log-msg">${escHtml(data.message || '')}</div>
      ${promptSnip}
    </div>`;

  entry.scrollIntoView({block:'nearest'});
}

function addPreview(path, shotId) {
  if (!path || previewsSeen.has(path)) return;
  previewsSeen.add(path);

  const section = document.getElementById('previewsSection');
  section.classList.add('visible');
  const grid = document.getElementById('previewsGrid');

  const card = document.createElement('div');
  card.className = 'preview-card' + (currentFormat === 'short' ? ' portrait' : '');
  card.title = shotId || path;
  card.onclick = () => window.open('/media?path=' + encodeURIComponent(path));
  card.innerHTML = `
    <img src="/media?path=${encodeURIComponent(path)}" alt="${escHtml(shotId)}" loading="lazy"
         onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
    <div class="pending" style="display:none">⏳</div>
    <div class="shot-label">${escHtml(shotId || '')}</div>`;
  grid.appendChild(card);
}

function showFinal(path) {
  const sec = document.getElementById('finalSection');
  sec.classList.add('visible');
  const v = document.getElementById('finalVideo');
  v.src = '/media?path=' + encodeURIComponent(path);
  v.load();
  const dl = document.getElementById('downloadBtn');
  dl.href = '/media?path=' + encodeURIComponent(path);
  dl.download = path.split('/').pop();
}

// ── Interactive Studio JS ───────────────────────────────────────────────────
function resolveMediaSrc(episodeDir, path) {
  if (!path) return '';
  if (path.startsWith('/') || path.includes(':\\') || path.includes(':/')) {
    return path;
  }
  return episodeDir + '/' + path;
}
let studioState = { active: false, current_step: 0, status: 'idle', shots: [] };
let studioPollInterval = null;

function studioStart() {
  const fmt = document.getElementById('studioFormat').value;
  fetch('/api/interactive/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({format: fmt})
  })
  .then(r => r.json())
  .then(data => {
    studioUpdateState(data);
    studioRunStep(1); // Auto run script gen when project starts
  });
}

function studioReset() {
  if (!confirm('Are you sure you want to discard this project? All generated files will be reset.')) return;
  fetch('/api/interactive/reset', {method: 'POST'})
  .then(r => r.json())
  .then(data => studioUpdateState(data));
}

function studioFetchState() {
  fetch('/api/interactive/state')
  .then(r => r.json())
  .then(data => studioUpdateState(data));
}

function studioUpdateState(data) {
  studioState = data;
  
  const setup = document.getElementById('studioSetup');
  const workspace = document.getElementById('studioWorkspace');
  
  if (!data.active) {
    setup.style.display = 'block';
    workspace.style.display = 'none';
    clearInterval(studioPollInterval);
    return;
  }
  
  setup.style.display = 'none';
  workspace.style.display = 'block';
  
  // Banner
  const banner = document.getElementById('studioStatusBanner');
  banner.className = 'studio-status-banner ' + data.status;
  document.getElementById('studioBannerMsg').textContent = data.message || 'Workspace idle';
  document.getElementById('studioBannerProgress').textContent = data.status === 'running' ? '⚡ Processing...' : '';
  
  // Update timeline accordion items
  for (let i = 1; i <= 6; i++) {
    const card = document.getElementById('stepCard' + i);
    const icon = document.getElementById('stepIcon' + i);
    
    if (i <= data.current_step) {
      card.className = 'timeline-step completed';
      icon.textContent = '✓ Done';
    } else if (i === data.current_step + 1) {
      card.className = 'timeline-step active';
      icon.textContent = data.status === 'running' ? '🔄 Running' : '⏳ Ready';
    } else {
      card.className = 'timeline-step';
      icon.textContent = '🔒 Locked';
    }
  }
  
  renderStepBodies(data);
  
  if (data.status === 'running') {
    if (!studioPollInterval) {
      studioPollInterval = setInterval(studioFetchState, 1500);
    }
  } else {
    clearInterval(studioPollInterval);
    studioPollInterval = null;
  }
}

function toggleStepBody(stepNum) {
  const card = document.getElementById('stepCard' + stepNum);
  if (card.classList.contains('active') || card.classList.contains('completed')) {
    const body = document.getElementById('stepBody' + stepNum);
    const isVisible = window.getComputedStyle(body).display !== 'none';
    body.style.display = isVisible ? 'none' : 'block';
  }
}

function studioRunStep(stepNum) {
  fetch('/api/interactive/run-step', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({step: stepNum})
  })
  .then(r => r.json())
  .then(() => {
    studioFetchState();
  });
}

function studioSaveAndContinue(stepNum) {
  studioSaveStepData(stepNum, () => {
    studioRunStep(stepNum + 1);
  });
}

function studioSaveStepData(stepNum, callback) {
  const container = document.getElementById('stepData' + stepNum);
  if (!container) return;
  
  const shotCards = container.getElementsByClassName('shot-editor-card');
  const shotsList = [];
  
  for (let card of shotCards) {
    const shotId = card.getAttribute('data-shot-id');
    const promptInput = card.querySelector('.prompt-input');
    const negInput = card.querySelector('.neg-input');
    const dialogueInput = card.querySelector('.dialogue-input');
    
    const shot = studioState.shots.find(s => s.shot_id === shotId) || { shot_id: shotId };
    
    if (promptInput) shot.image_prompt = promptInput.value;
    if (negInput) shot.negative_prompt = negInput.value;
    if (dialogueInput && shot.dialogue && shot.dialogue[0]) {
      shot.dialogue[0].line = dialogueInput.value;
    }
    
    shotsList.push(shot);
  }
  
  fetch('/api/interactive/save-script', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({shots: shotsList})
  })
  .then(r => r.json())
  .then(data => {
    if (callback) callback();
    else alert('✓ Step data saved!');
  });
}

function studioRegenerateShotImage(shotId) {
  const card = document.querySelector(`.shot-editor-card[data-shot-id="${shotId}"]`);
  const prompt = card.querySelector('.prompt-input').value;
  const neg = card.querySelector('.neg-input').value;
  
  fetch('/api/interactive/regenerate-shot-image', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({shot_id: shotId, prompt: prompt, negative_prompt: neg})
  })
  .then(() => {
    alert('🎨 Image generation started on GPU. Please wait...');
    studioFetchState();
  });
}

function studioRegenerateShotVideo(shotId) {
  const card = document.querySelector(`.shot-editor-card[data-shot-id="${shotId}"]`);
  const prompt = card.querySelector('.prompt-input').value;
  const neg = card.querySelector('.neg-input').value;
  
  fetch('/api/interactive/regenerate-shot-video', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({shot_id: shotId, prompt: prompt, negative_prompt: neg})
  })
  .then(() => {
    alert('🎬 Animation generation started on GPU. Please wait...');
    studioFetchState();
  });
}

function renderStepBodies(data) {
  const step = data.current_step;
  const shots = data.shots || [];
  
  // Render Step 1
  if (data.current_step >= 0) {
    const div = document.getElementById('stepData1');
    if (shots.length === 0) {
      div.innerHTML = `<p style="color:var(--text3);text-align:center;padding:20px">No script generated yet. Click "Generate Script" above.</p>`;
    } else {
      let html = '';
      shots.forEach(s => {
        const line = (s.dialogue && s.dialogue[0]) ? s.dialogue[0].line : '';
        html += `
          <div class="shot-editor-card" data-shot-id="${s.shot_id}">
            <h4 style="font-family:'JetBrains Mono';font-size:12px;color:var(--blue)">Shot: ${s.shot_id} (${s.character || 'Narrator'})</h4>
            <div class="shot-editor-grid">
              <div class="editor-fields-col">
                <div class="editor-field">
                  <label>Dialogue Line (Hindi / English)</label>
                  <textarea class="editor-textarea dialogue-input">${escHtml(line)}</textarea>
                </div>
                <div class="editor-field">
                  <label>Image Generation Prompt</label>
                  <input type="text" class="editor-input prompt-input" value="${escHtml(s.image_prompt || '')}">
                </div>
                <div class="editor-field">
                  <label>Negative Prompt</label>
                  <input type="text" class="editor-input neg-input" value="${escHtml(s.negative_prompt || '')}">
                </div>
              </div>
              <div class="editor-preview-col" style="color:var(--text3);font-size:12px">
                📝 Modify prompts or dialogue. Click Save.
              </div>
            </div>
            <div style="text-align:right">
              <button class="action-btn secondary" onclick="studioSaveStepData(1)" style="padding:6px 12px;font-size:11px">💾 Save Shot</button>
            </div>
          </div>`;
      });
      div.innerHTML = html;
    }
  }
  
  // Render Step 2: TTS players
  if (data.current_step >= 1) {
    const div = document.getElementById('stepData2');
    let html = '';
    shots.forEach(s => {
      const line = (s.dialogue && s.dialogue[0]) ? s.dialogue[0].line : '';
      const audioPath = (s.dialogue && s.dialogue[0]) ? s.dialogue[0].audio_path : '';
      const audioPlayer = audioPath 
        ? `<audio controls src="/media?path=${encodeURIComponent(data.episode_dir + '/' + audioPath)}" style="width:100%;height:32px;margin-top:8px"></audio>`
        : `<span style="color:var(--text3)">No dialogue audio</span>`;
        
      html += `
        <div class="shot-editor-card" data-shot-id="${s.shot_id}">
          <h4 style="font-family:'JetBrains Mono';font-size:12px;color:var(--blue)">Shot: ${s.shot_id} (${s.character || 'Narrator'})</h4>
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
            <span style="font-style:italic;color:var(--text2)">"${escHtml(line || 'B-roll/No speech')}"</span>
            <div style="min-width:240px">${audioPlayer}</div>
          </div>
        </div>`;
    });
    div.innerHTML = html || `<p style="color:var(--text3)">No audio clips generated yet.</p>`;
  }
  
  // Render Step 3: Images & single-shot regen
  if (data.current_step >= 2) {
    const div = document.getElementById('stepData3');
    let html = '';
    shots.forEach(s => {
      const imgPath = resolveMediaSrc(data.episode_dir, s.scene_image);
      const isPortrait = data.format === 'short';
      const imgTag = imgPath
        ? `<img class="shot-preview-img ${isPortrait ? 'portrait' : ''}" src="/media?path=${encodeURIComponent(imgPath)}" alt="Scene image">`
        : `<div class="pending" style="font-size:32px">🎨</div>`;
        
      html += `
        <div class="shot-editor-card" data-shot-id="${s.shot_id}">
          <h4 style="font-family:'JetBrains Mono';font-size:12px;color:var(--blue)">Shot: ${s.shot_id}</h4>
          <div class="shot-editor-grid">
            <div class="editor-fields-col">
              <div class="editor-field">
                <label>Stable Diffusion Prompt</label>
                <textarea class="editor-textarea prompt-input">${escHtml(s.image_prompt || '')}</textarea>
              </div>
              <div class="editor-field">
                <label>Negative Prompt</label>
                <input type="text" class="editor-input neg-input" value="${escHtml(s.negative_prompt || '')}">
              </div>
              <div style="margin-top:8px">
                <button class="action-btn primary" onclick="studioRegenerateShotImage('${s.shot_id}')" style="padding:6px 12px;font-size:11px">🎨 Regenerate Image</button>
              </div>
            </div>
            <div class="editor-preview-col">
              ${imgTag}
            </div>
          </div>
        </div>`;
    });
    div.innerHTML = html || `<p style="color:var(--text3)">No images generated yet.</p>`;
  }
  
  // Render Step 4: Upscaled view
  if (data.current_step >= 3) {
    const div = document.getElementById('stepData4');
    let html = '<div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(180px, 1fr));gap:12px">';
    shots.forEach(s => {
      const imgPath = resolveMediaSrc(data.episode_dir, s.scene_image_upscaled || s.scene_image);
      const isPortrait = data.format === 'short';
      html += `
        <div style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;overflow:hidden;padding:8px">
          <div style="font-family:'JetBrains Mono';font-size:11px;margin-bottom:6px;font-weight:600">${s.shot_id}</div>
          <img src="/media?path=${encodeURIComponent(imgPath)}" style="width:100%;aspect-ratio:${isPortrait ? '9/16' : '16/9'};object-fit:cover;border-radius:4px;border:1px solid var(--border)">
        </div>`;
    });
    html += '</div>';
    div.innerHTML = html;
  }
  
  // Render Step 5: Animation videos & single-shot regen
  if (data.current_step >= 4) {
    const div = document.getElementById('stepData5');
    let html = '';
    shots.forEach(s => {
      const clipPath = resolveMediaSrc(data.episode_dir, s.animated_clip);
      const isPortrait = data.format === 'short';
      const videoTag = clipPath
        ? `<video class="shot-preview-video ${isPortrait ? 'portrait' : ''}" src="/media?path=${encodeURIComponent(clipPath)}" controls loop autoplay muted></video>`
        : `<div class="pending" style="font-size:32px">🎬</div>`;
        
      html += `
        <div class="shot-editor-card" data-shot-id="${s.shot_id}">
          <h4 style="font-family:'JetBrains Mono';font-size:12px;color:var(--blue)">Shot: ${s.shot_id}</h4>
          <div class="shot-editor-grid">
            <div class="editor-fields-col">
              <div class="editor-field">
                <label>AnimateDiff Prompt (motion direction, speed, features)</label>
                <textarea class="editor-textarea prompt-input">${escHtml(s.image_prompt || '')}</textarea>
              </div>
              <div class="editor-field">
                <label>Negative Prompt</label>
                <input type="text" class="editor-input neg-input" value="${escHtml(s.negative_prompt || '')}">
              </div>
              <div style="margin-top:8px">
                <button class="action-btn primary" onclick="studioRegenerateShotVideo('${s.shot_id}')" style="padding:6px 12px;font-size:11px">🎬 Regenerate Video</button>
              </div>
            </div>
            <div class="editor-preview-col">
              ${videoTag}
            </div>
          </div>
        </div>`;
    });
    div.innerHTML = html || `<p style="color:var(--text3)">No animation clips generated yet.</p>`;
  }
  
  // Render Step 6: Final assembled mp4 player
  if (data.current_step >= 5) {
    const div = document.getElementById('stepData6');
    const finalMp4 = data.episode_dir + '/assembled.mp4';
    
    div.innerHTML = `
      <div style="background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius-sm);padding:20px;text-align:center">
        <video src="/media?path=${encodeURIComponent(finalMp4)}" controls style="width:100%;max-width:640px;margin-bottom:16px"></video>
        <div>
          <a class="action-btn primary" style="display:inline-flex" href="/media?path=${encodeURIComponent(finalMp4)}" download>⬇ Download Assembled Video (2K MP4)</a>
        </div>
      </div>`;
  }
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            self._html(HTML)
        elif path == "/api/progress-stream":
            self._sse_stream()
        elif path == "/api/interactive/state":
            self._interactive_state_get()
        elif path.startswith("/media"):
            self._serve_media()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if path == "/api/run":
            self._start_pipeline(data.get("format", "full"))
        elif path == "/api/stop":
            self._stop_pipeline()
        elif path == "/api/interactive/start":
            self._interactive_start(data.get("format", "full"))
        elif path == "/api/interactive/reset":
            self._interactive_reset()
        elif path == "/api/interactive/save-script":
            self._interactive_save_script(data.get("shots", []))
        elif path == "/api/interactive/run-step":
            self._interactive_run_step(data.get("step", 1))
        elif path == "/api/interactive/regenerate-shot-image":
            self._interactive_regenerate_shot_image(data)
        elif path == "/api/interactive/regenerate-shot-video":
            self._interactive_regenerate_shot_video(data)
        else:
            self._json(404, {"error": "not found"})

    # ── Serve HTML ─────────────────────────────────────────────────────────
    def _html(self, content: str):
        enc = content.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(enc))
        self.end_headers()
        self.wfile.write(enc)

    # ── Serve media (images, videos) ──────────────────────────────────────
    def _serve_media(self):
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        media_path = qs.get("path", [""])[0]
        p = Path(media_path)
        if not p.exists() or not p.is_file():
            self._json(404, {"error": "file not found"})
            return
        mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    # ── SSE progress stream ────────────────────────────────────────────────
    def _sse_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_state = {}
        while True:
            try:
                progress = _read_progress()
                with _job_lock:
                    state = dict(_job)

                # Merge progress file into state
                if progress:
                    state.update({
                        "percent": progress.get("percent", state["percent"]),
                        "stage":   progress.get("stage", state["stage"]),
                        "message": progress.get("message", state["message"]),
                        "status":  progress.get("status", state["status"]),
                    })

                # Scan for new scene preview images
                state["previews"] = _scan_previews(state.get("format", ""))

                # Also check for final video
                if not state.get("final_path"):
                    state["final_path"] = _find_latest_final()

                if state != last_state:
                    payload = json.dumps(state, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    last_state = dict(state)

                if state.get("status") in ("done", "failed"):
                    break

                time.sleep(0.8)
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception:
                break

    # ── Start pipeline ─────────────────────────────────────────────────────
    def _start_pipeline(self, fmt: str):
        global _job_proc
        with _job_lock:
            if _job["status"] == "running":
                self._json(409, {"error": "pipeline already running"})
                return
            _job.update({
                "status": "running", "format": fmt, "percent": 0.0,
                "stage": "bootstrap", "message": "Starting…",
                "log": [], "previews": [], "final_path": "", "error": ""
            })

        UI_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        UI_PROGRESS_FILE.write_text(
            json.dumps({"percent": 0, "stage": "bootstrap", "message": "starting", "status": "running"}),
            encoding="utf-8"
        )

        def _worker():
            global _job_proc
            env = os.environ.copy()
            env["2DVIDEO_PROGRESS_FILE"] = str(UI_PROGRESS_FILE)
            cmd = [
                sys.executable, str(ROOT / "pipeline" / "orchestrator.py"),
                f"--format={fmt}",
                "--use-ollama",
                "--kind=poem" if fmt == "short" else "--kind=story",
            ]
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            _job_proc = proc

            out_lines: list[str] = []
            if proc.stdout:
                for line in proc.stdout:
                    out_lines.append(line.rstrip())

            rc = proc.wait()
            final = _find_latest_final()
            with _job_lock:
                _job["status"] = "done" if rc == 0 else "failed"
                _job["final_path"] = final or ""
                if rc != 0:
                    _job["error"] = "\n".join(out_lines[-20:])

            if rc != 0:
                _write_progress_file(
                    percent=_job["percent"],
                    stage=_job["stage"],
                    message="\n".join(out_lines[-5:]),
                    status="failed",
                )
            else:
                _write_progress_file(percent=100.0, stage="complete",
                                     message="Pipeline complete ✓", status="done")

        threading.Thread(target=_worker, daemon=True).start()
        self._json(200, {"ok": True, "format": fmt})

    # ── Stop pipeline ──────────────────────────────────────────────────────
    def _stop_pipeline(self):
        global _job_proc
        if _job_proc and _job_proc.poll() is None:
            _job_proc.terminate()
        with _job_lock:
            _job["status"] = "idle"
        self._json(200, {"ok": True})

    def _json(self, code: int, data: dict):
        enc = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(enc))
        self.end_headers()
        self.wfile.write(enc)

    # ── Interactive Studio Endpoints ──────────────────────────────────────────
    def _interactive_state_get(self):
        global _interactive_state
        if _interactive_state["active"]:
            # Load the most recent generated JSON file to populate shots list
            work_dir = Path(_interactive_state["episode_dir"])
            stage_files = [
                work_dir / "episode.json",          # step 0 / 1 input
                work_dir / "episode.json",          # step 1 output
                work_dir / "episode.tts.json",      # step 2 output
                work_dir / "episode.scenes.json",   # step 3 output
                work_dir / "episode.upscaled.json", # step 4 output
                work_dir / "episode.animated.json", # step 5 output
            ]
            step = _interactive_state["current_step"]
            shots = []
            for i in range(step, -1, -1):
                if i < len(stage_files):
                    f = stage_files[i]
                    sys.stderr.write(f"[ui] GET STATE checking file {i} path={f} exists={f.exists()}\n")
                    sys.stderr.flush()
                    if f.exists():
                        try:
                            data = json.loads(f.read_text(encoding="utf-8"))
                            shots = data.get("shots", [])
                            sys.stderr.write(f"[ui] GET STATE successfully loaded shots len={len(shots)}\n")
                            sys.stderr.flush()
                            break
                        except Exception as e:
                            sys.stderr.write(f"[ui] GET STATE exception={e}\n")
                            sys.stderr.flush()
            _interactive_state["shots"] = shots
        sys.stderr.write(f"[ui] GET STATE id={id(_interactive_state)} active={_interactive_state.get('active')} status={_interactive_state.get('status')} current_step={_interactive_state.get('current_step')} shots_len={len(_interactive_state.get('shots', []))}\n")
        sys.stderr.flush()
        self._json(200, _interactive_state)

    def _interactive_start(self, fmt: str):
        global _interactive_state
        ep_id = f"interactive-{int(time.time())}"
        ep_dir = OUTPUT_DIR / "interactive" / ep_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        
        _interactive_state.update({
            "active": True,
            "format": fmt,
            "episode_id": ep_id,
            "current_step": 0,
            "status": "idle",
            "message": "Ready to generate script",
            "error": "",
            "episode_dir": str(ep_dir),
            "shots": []
        })
        
        # Write skeleton episode.json
        skeleton = {"episode_id": ep_id, "format": fmt, "shots": []}
        write_json(ep_dir / "episode.json", skeleton)
        
        self._json(200, _interactive_state)

    def _interactive_reset(self):
        global _interactive_state
        _interactive_state.update({
            "active": False,
            "format": "full",
            "episode_id": "",
            "current_step": 0,
            "status": "idle",
            "message": "",
            "error": "",
            "episode_dir": "",
            "shots": []
        })
        self._json(200, _interactive_state)

    def _interactive_save_script(self, shots_list: list):
        global _interactive_state
        if not _interactive_state["active"]:
            self._json(400, {"error": "No active interactive session"})
            return
            
        work_dir = Path(_interactive_state["episode_dir"])
        step = _interactive_state["current_step"]
        
        filenames = [
            "episode.json",          # step 0
            "episode.json",          # step 1
            "episode.tts.json",      # step 2
            "episode.scenes.json",   # step 3
            "episode.upscaled.json", # step 4
            "episode.animated.json", # step 5
        ]
        
        target_file = work_dir / filenames[step]
        if target_file.exists():
            try:
                data = json.loads(target_file.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}
            
        data["shots"] = shots_list
        write_json(target_file, data)
        _interactive_state["shots"] = shots_list
        self._json(200, {"ok": True, "shots": shots_list})

    def _interactive_run_step(self, step_num: int):
        global _interactive_state
        if not _interactive_state["active"]:
            self._json(400, {"error": "No active interactive session"})
            return
            
        if _interactive_state["status"] == "running":
            self._json(409, {"error": "Interactive step already running"})
            return
            
        _interactive_state["status"] = "running"
        _interactive_state["message"] = f"Running step {step_num}…"
        _interactive_state["error"] = ""
        
        work_dir = Path(_interactive_state["episode_dir"])
        fmt = _interactive_state["format"]
        
        kind = "poem" if fmt == "short" else "story"
        template = "config/episode_templates/poem_template.json" if kind == "poem" else "config/episode_templates/story_template.json"
        
        step_cmds = {
            1: [
                sys.executable, str(ROOT / "pipeline" / "01_story_engine.py"),
                "--template", template,
                "--output", str(work_dir / "episode.json"),
                "--work-dir", str(work_dir),
                "--episode-id", _interactive_state["episode_id"],
                "--kind", kind,
                "--format", fmt,
                "--use-ollama"
            ],
            2: [sys.executable, str(ROOT / "pipeline" / "03_tts.py"), "--episode", str(work_dir / "episode.json"), "--output", str(work_dir / "episode.tts.json")],
            3: [sys.executable, str(ROOT / "pipeline" / "04_scene_gen.py"), "--episode", str(work_dir / "episode.tts.json"), "--work-dir", str(work_dir), "--output", str(work_dir / "episode.scenes.json"), "--format", fmt],
            4: [sys.executable, str(ROOT / "pipeline" / "06_upscale.py"), "--episode", str(work_dir / "episode.scenes.json"), "--work-dir", str(work_dir / "scenes_upscaled"), "--output", str(work_dir / "episode.upscaled.json"), "--no-realesrgan"],
            5: [sys.executable, str(ROOT / "pipeline" / "06_animatediff.py"), "--episode", str(work_dir / "episode.upscaled.json"), "--work-dir", str(work_dir / "animated"), "--output", str(work_dir / "episode.animated.json")],
            6: [sys.executable, str(ROOT / "pipeline" / "07_assemble.py"), "--episode", str(work_dir / "episode.animated.json"), "--output", str(work_dir / "assembled.mp4")]
        }
        
        cmd = step_cmds.get(step_num)
        if not cmd:
            self._json(400, {"error": f"Invalid step number {step_num}"})
            return
            
        def _worker():
            global _interactive_state
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"/home/lalit/Desktop/GPU optimization/Wan2GP/ffmpeg_bins:{env.get('LD_LIBRARY_PATH', '')}"
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            out_lines = []
            if proc.stdout:
                for line in proc.stdout:
                    out_lines.append(line.rstrip())
                    _interactive_state["message"] = line.rstrip()
            rc = proc.wait()
            
            if rc == 0:
                if step_num == 1:
                    try:
                        from pipeline.prompt_engine import enrich_episode_with_prompts
                        ep_path = work_dir / "episode.json"
                        if ep_path.exists():
                            ep_data = json.loads(ep_path.read_text(encoding="utf-8"))
                            ep_data = enrich_episode_with_prompts(ep_data, format=fmt, use_ollama=True)
                            write_json(ep_path, ep_data)
                    except Exception as e:
                        log.warning(f"[ui] Prompt enrichment failed: {e}")
                sys.stderr.write(f"[ui] WORKER SUCCESS step={step_num} setting status to idle id={id(_interactive_state)}\n")
                sys.stderr.flush()
                _interactive_state["status"] = "idle"
                _interactive_state["current_step"] = step_num
                _interactive_state["message"] = f"Step {step_num} complete! ✓"
            else:
                sys.stderr.write(f"[ui] WORKER FAILED step={step_num} rc={rc}\n")
                sys.stderr.flush()
                _interactive_state["status"] = "failed"
                _interactive_state["message"] = f"Step {step_num} failed"
                _interactive_state["error"] = "\n".join(out_lines[-20:])
                
        threading.Thread(target=_worker, daemon=True).start()
        self._json(200, {"ok": True, "step": step_num})

    def _interactive_regenerate_shot_image(self, data: dict):
        global _interactive_state
        if not _interactive_state["active"]:
            self._json(400, {"error": "No active session"})
            return
            
        shot_id = data.get("shot_id")
        prompt = data.get("prompt")
        neg_prompt = data.get("negative_prompt")
        
        if not shot_id or not prompt:
            self._json(400, {"error": "Missing shot_id or prompt"})
            return
            
        _interactive_state["status"] = "running"
        _interactive_state["message"] = f"Regenerating image for {shot_id}…"
        
        def _worker():
            global _interactive_state
            try:
                work_dir = Path(_interactive_state["episode_dir"])
                episode_path = work_dir / "episode.scenes.json"
                if not episode_path.exists():
                    episode_path = work_dir / "episode.tts.json"
                
                from pipeline.sd_client import generate_image, dims_for_format
                
                with open(episode_path, "r", encoding="utf-8") as f:
                    episode = json.load(f)
                    
                shot = next((s for s in episode["shots"] if s["shot_id"] == shot_id), None)
                if not shot:
                    raise ValueError(f"Shot {shot_id} not found")
                    
                w, h = dims_for_format(_interactive_state["format"])
                img_path = Path(shot.get("scene_image") or f"scenes/scenes/{shot_id}.png")
                if not img_path.is_absolute():
                    img_path = work_dir / img_path
                    
                img_path.parent.mkdir(parents=True, exist_ok=True)
                
                generate_image(prompt=prompt, negative_prompt=neg_prompt, width=w, height=h, steps=22, output_path=img_path)
                
                shot["image_prompt"] = prompt
                shot["negative_prompt"] = neg_prompt
                shot["scene_image"] = str(img_path.relative_to(work_dir) if img_path.is_relative_to(work_dir) else img_path)
                
                with open(episode_path, "w", encoding="utf-8") as f:
                    json.dump(episode, f, indent=2, ensure_ascii=False)
                    
                _interactive_state["status"] = "idle"
                _interactive_state["message"] = f"✓ {shot_id} image regenerated successfully!"
            except Exception as e:
                _interactive_state["status"] = "failed"
                _interactive_state["message"] = f"Failed to regenerate image: {e}"
                _interactive_state["error"] = str(e)
                
        threading.Thread(target=_worker, daemon=True).start()
        self._json(200, {"ok": True})

    def _interactive_regenerate_shot_video(self, data: dict):
        global _interactive_state
        if not _interactive_state["active"]:
            self._json(400, {"error": "No active session"})
            return
            
        shot_id = data.get("shot_id")
        prompt = data.get("prompt")
        neg_prompt = data.get("negative_prompt")
        
        if not shot_id or not prompt:
            self._json(400, {"error": "Missing shot_id or prompt"})
            return
            
        _interactive_state["status"] = "running"
        _interactive_state["message"] = f"Regenerating video for {shot_id}…"
        
        def _worker():
            global _interactive_state
            try:
                work_dir = Path(_interactive_state["episode_dir"])
                episode_path = work_dir / "episode.animated.json"
                if not episode_path.exists():
                    episode_path = work_dir / "episode.upscaled.json"
                    
                import importlib
                animatediff_module = importlib.import_module("pipeline.06_animatediff")
                _load_pipeline = animatediff_module._load_pipeline
                _animate_shot = animatediff_module._animate_shot
                
                with open(episode_path, "r", encoding="utf-8") as f:
                    episode = json.load(f)
                    
                shot = next((s for s in episode["shots"] if s["shot_id"] == shot_id), None)
                if not shot:
                    raise ValueError(f"Shot {shot_id} not found")
                    
                image_path = Path(shot.get("scene_image_upscaled") or shot.get("scene_image", ""))
                if not image_path.is_absolute():
                    image_path = work_dir / image_path
                    
                output_path = Path(shot.get("animated_clip") or f"animated/animated/{shot_id}.mp4")
                if not output_path.is_absolute():
                    output_path = work_dir / output_path
                    
                device = "cuda"
                pipe = _load_pipeline(device)
                _animate_shot(
                    pipe,
                    image_path=image_path,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    output_path=output_path,
                    frames=16,
                    fps=8,
                    steps=4,
                    device=device
                )
                
                shot["image_prompt"] = prompt
                shot["negative_prompt"] = neg_prompt
                shot["animated_clip"] = str(output_path.relative_to(work_dir) if output_path.is_relative_to(work_dir) else output_path)
                
                with open(episode_path, "w", encoding="utf-8") as f:
                    json.dump(episode, f, indent=2, ensure_ascii=False)
                    
                _interactive_state["status"] = "idle"
                _interactive_state["message"] = f"✓ {shot_id} video regenerated successfully!"
            except Exception as e:
                _interactive_state["status"] = "failed"
                _interactive_state["message"] = f"Failed to regenerate video: {e}"
                _interactive_state["error"] = str(e)
                
        threading.Thread(target=_worker, daemon=True).start()
        self._json(200, {"ok": True})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_progress() -> dict:
    if not UI_PROGRESS_FILE.exists():
        return {}
    try:
        return json.loads(UI_PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_progress_file(*, percent: float, stage: str, message: str, status: str):
    UI_PROGRESS_FILE.write_text(
        json.dumps({"percent": percent, "stage": stage, "message": message, "status": status}),
        encoding="utf-8",
    )


def _scan_previews(fmt: str = "") -> list[dict]:
    """Find all scene PNGs generated so far across today's episode dirs."""
    previews: list[dict] = []
    for scenes_dir in OUTPUT_DIR.rglob("scenes"):
        if not scenes_dir.is_dir():
            continue
        for png in sorted(scenes_dir.glob("s*.png")):
            if "_upscaled" in png.name:
                continue
            shot_id = png.stem
            previews.append({"shot_id": shot_id, "path": str(png)})
    return previews


def _find_latest_final() -> str:
    """Find the most recently created final.mp4 or assembled.mp4."""
    candidates = sorted(OUTPUT_DIR.rglob("assembled.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    return ""


# ── Entry point ───────────────────────────────────────────────────────────────

def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"\n🍎 Fruit Cartoon Studio")
    print(f"   → http://{host}:{port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = p.parse_args()
    main(a.host, a.port)
