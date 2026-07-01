"""
sd_client.py — stable-diffusion.cpp REST API wrapper
------------------------------------------------------
Manages the lifecycle of sd-server-vulkan and exposes
a clean generate_image() call used by 04_scene_gen.py.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
import urllib.request
import urllib.error
from base64 import b64decode
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SD_STUDIO = Path("/home/lalit/Desktop/GPU optimization/Optimized_AI Studio")
SD_BACKEND = SD_STUDIO / "app/backend/linux/vulkan"
SD_SERVER_BIN = SD_BACKEND / "sd-server-vulkan"
SD_LIB_DIR = str(SD_BACKEND)

MODELS_DIR = SD_STUDIO / "app/models"
# Best model for kids cartoon style at 20-25 steps: DreamShaperXL_Lightning
# (Despite the "Lightning" name, at 20-25 steps with euler_a it produces
#  rich, stylised cartoon output — better than photorealistic Juggernaut for
#  our Cocomelon aesthetic)
DEFAULT_MODEL = MODELS_DIR / "DreamShaperXL_Lightning.safetensors"

SD_PORT = 18765  # Internal port — not exposed to user
SD_URL = f"http://127.0.0.1:{SD_PORT}"

_server_proc: subprocess.Popen | None = None


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_server(
    model_path: Path | None = None,
    *,
    port: int = SD_PORT,
    wait_sec: int = 240,
) -> None:
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        log.info("[sd_client] sd-server-vulkan already running (pid=%s)", _server_proc.pid)
        return

    model = model_path or DEFAULT_MODEL
    if not model.exists():
        raise FileNotFoundError(f"[sd_client] Model not found: {model}")
    if not SD_SERVER_BIN.exists():
        raise FileNotFoundError(f"[sd_client] sd-server-vulkan not found: {SD_SERVER_BIN}")

    cmd = [
        str(SD_SERVER_BIN),
        "--model", str(model),
        "--listen-port", str(port),
        "--listen-ip", "127.0.0.1",
        "--threads", "-1",       # auto CPU threads
        "--mmap",                # memory-map model for instant load
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{SD_LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}"

    log.info("[sd_client] Starting sd-server-vulkan: %s", " ".join(cmd))
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sd_server.log"
    log_fh = open(log_file, "w", encoding="utf-8")

    _server_proc = subprocess.Popen(
        cmd, env=env,
        stdout=log_fh, stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for ready
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if _server_proc.poll() is not None:
            log_fh.close()
            out = log_file.read_text(encoding="utf-8")[-1000:]
            raise RuntimeError(f"[sd_client] sd-server-vulkan exited early.\n{out}")
        try:
            with urllib.request.urlopen(f"{SD_URL}/", timeout=2):
                log.info("[sd_client] sd-server-vulkan ready on port %s", port)
                return
        except Exception:
            time.sleep(1)

    raise TimeoutError(f"[sd_client] sd-server-vulkan did not become ready in {wait_sec}s")


def stop_server() -> None:
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        log.info("[sd_client] Stopping sd-server-vulkan (pid=%s)", _server_proc.pid)
        try:
            _server_proc.send_signal(signal.SIGTERM)
            _server_proc.wait(timeout=10)
        except Exception:
            _server_proc.kill()
        _server_proc = None


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_image(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 576,
    steps: int = 22,
    cfg_scale: float = 7.0,
    seed: int = -1,
    output_path: Path,
    sampling_method: str = "euler_a",
) -> Path:
    """
    POST /txt2img to the running sd-server-vulkan and save result PNG.
    Returns the output_path on success.
    """
    payload = json.dumps({
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sample_method": sampling_method,
        "batch_count": 1,
        "batch_size": 1,
    }).encode()

    req = urllib.request.Request(
        f"{SD_URL}/sdapi/v1/txt2img",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"[sd_client] /sdapi/v1/txt2img HTTP {exc.code}: {body}") from exc

    elapsed = time.time() - t0
    log.info("[sd_client] Generated in %.1fs → %s", elapsed, output_path.name)

    # The server returns {"images": ["<base64png>"]}
    images = data.get("images", [])
    if not images:
        raise RuntimeError(f"[sd_client] No images in response: {data}")

    img_bytes = b64decode(images[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(img_bytes)
    return output_path


# ---------------------------------------------------------------------------
# Convenience: dimensions for each format
# ---------------------------------------------------------------------------

def dims_for_format(fmt: str) -> tuple[int, int]:
    """Returns (width, height) for the given format string."""
    if fmt == "short":
        return 576, 1024   # 9:16 portrait
    return 1024, 576       # 16:9 landscape (default)
