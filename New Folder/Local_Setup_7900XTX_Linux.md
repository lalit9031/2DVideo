# RUNNING THE FRUIT CARTOON PIPELINE LOCALLY
### Hardware: AMD Radeon RX 7900 XTX (24GB VRAM) + 64GB RAM + Linux
### Goal: Run script generation, character/environment images, video clips, and Hindi voice generation entirely on your own machine — no cloud APIs.

---

## REALITY CHECK FIRST

Your hardware is actually well-suited for this:
- **24GB VRAM** on the 7900 XTX is enough for SDXL/Flux image generation and WAN 2.2 / LTX video generation at quantized precision.
- **64GB system RAM** gives you plenty of headroom for model offloading and running an LLM + TTS alongside image/video tools.
- AMD's ROCm has matured a lot — as of mid-2026, ROCm 7.2.x officially supports your GPU (gfx1100/RDNA3) and Ollama, ComfyUI, llama.cpp, and vLLM all run natively.

What you're building locally, mapped to the pipeline phases:

| Pipeline phase | Local tool |
|---|---|
| Phase 1–2 (topics, script) | Local LLM via Ollama |
| Phase 3–5 (character/environment/scene images) | ComfyUI + SDXL or Flux |
| Phase 6 (video clips) | ComfyUI + WAN 2.2 or LTX Video |
| Hindi dialogue voice | Coqui XTTS-v2 (voice cloning, supports Hindi) or AI4Bharat Indic-TTS |
| Phase 7 (metadata) | Local LLM via Ollama |

---

## STEP 1 — Install ROCm

```bash
# Ubuntu 24.04 recommended (best-tested ROCm target)
sudo apt update
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_latest_all.deb
sudo apt install ./amdgpu-install_latest_all.deb
sudo amdgpu-install --usecase=rocm,hiplibsdk -y
sudo usermod -aG render,video $USER
# reboot after this
sudo reboot
```

Verify it worked:
```bash
rocminfo | grep gfx
# should show gfx1100 (your 7900 XTX)
```

---

## STEP 2 — Install Ollama (for script/topic/metadata generation — Phases 1, 2, 7)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
# or a stronger model if you want better Hindi dialogue quality:
ollama pull qwen2.5:14b
```

Ollama auto-detects ROCm on your 7900 XTX. Test it:
```bash
ollama run qwen2.5:14b "Write a 2-line children's story moral about honesty."
```

**Use this model to run Phases 1, 2, and 7** of your pipeline — feed it the master prompt (your uploaded Video_Master_Prompt.txt) plus the 4 Bible files as context, and it can generate topics, scripts, and metadata locally.

---

## STEP 3 — Install ComfyUI (for Phases 3, 4, 5 — image generation)

Docker is the most reliable path on your GPU:

```bash
WORKSPACE_DIR="$HOME/comfyui_workspace"
mkdir -p $WORKSPACE_DIR

docker pull rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.6.0

docker run -it \
  --name comfyui-rocm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --network=host \
  -v $WORKSPACE_DIR:/workspace \
  -w /workspace \
  rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  /bin/bash
```

Inside the container:
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
python main.py --listen 0.0.0.0
```

Open `http://127.0.0.1:8188` in your browser.

**Model to download for character/environment images:** SDXL (fits comfortably in 24GB) or Flux Dev in FP8/GGUF quantized form (Flux BF16 alone needs ~24GB, too tight — use quantized).

**Use this for Phase 3 (characters) and Phase 4 (environments):** paste the LOCKED IMAGE PROMPT from your Character Bible or Environment Bible directly into ComfyUI's positive prompt node.

---

## STEP 4 — Video generation (Phase 6)

ComfyUI supports two solid local video models that fit your VRAM:

- **WAN 2.2** — best current all-rounder for image-to-video, runs well on 7900 XTX with the flash-attention tuning below.
- **LTX Video** — faster/lighter, good for quick iteration.

Download the model + text encoder into `ComfyUI/models/checkpoints` and `ComfyUI/models/text_encoders`, then load a WAN 2.2 or LTX text-to-video/image-to-video workflow template from ComfyUI's built-in browser (Workflow → Browse Templates → Video).

**Known issue to watch for:** some users hit stalls loading LTX 2.3 on 7900 XTX/ROCm unless dynamic VRAM, pinned memory, and async offload are disabled. If you see ComfyUI hang on "Requested to load LTXAV," launch with:
```bash
python main.py --disable-smart-memory
```
as a first troubleshooting step, or switch to WAN 2.2 which has fewer reported issues on your card.

**Performance tuning (optional but recommended for your gfx1100 card):**
```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON='{"BLOCK_M":128,"BLOCK_N":64,"waves_per_eu":1,"PRE_LOAD_V":false,"num_stages":1,"num_warps":8}'
export PYTORCH_MIOPEN_SUGGEST_NHWC=0
export COMFYUI_ENABLE_MIOPEN=1
export MIOPEN_FIND_MODE=FAST
```
This significantly speeds up video model attention layers on RDNA3.

**Use this for Phase 6:** feed in your merged video prompt (Environment + Character + Acting Guide + Voice Bible combined, as described in the Master Workflow) as the text conditioning, and use the Phase 3 character image as the reference/init image for image-to-video.

---

## STEP 5 — Hindi voice generation (dialogue audio)

Two good local options:

**Option A — Coqui XTTS-v2** (voice cloning, supports Hindi among 17 languages, only needs a 6-second reference clip per character voice):
```bash
pip install TTS
```
```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")  # ROCm maps to "cuda" device name in PyTorch
tts.tts_to_file(
    text="नमस्ते, मैं Appy हूँ।",
    speaker_wav="appy_reference_voice.wav",
    language="hi",
    file_path="output_appy_line1.wav"
)
```
Record or generate one short reference clip per character (matching the tone described in your Voice Bible) and reuse it for every line that character speaks — this gives you consistent character voices across all future videos, the same way the Character Bible keeps their look consistent.

**Option B — AI4Bharat Indic-TTS** — purpose-built for Hindi and 12 other Indian languages, better native Hindi prosody than XTTS but no voice cloning (fixed trained voices). Good as a fallback or for background/crowd voices.

---

## STEP 6 — Tie it all together with a simple automation script

A basic Python orchestrator can chain: Ollama (script) → ComfyUI API (images/video) → XTTS (audio). ComfyUI has a REST API (`/prompt` endpoint) so you can submit workflows programmatically instead of clicking through the UI each time.

Rough shape:
```python
import requests, json

# 1. Generate script via Ollama
script = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5:14b",
    "prompt": open("Video_Master_Prompt.txt").read() + "\n\nGenerate Phase 2 script for topic 3."
}).json()

# 2. Submit character image workflow to ComfyUI
with open("character_workflow.json") as f:
    workflow = json.load(f)
# inject the locked prompt from Character Bible into the workflow's prompt node
requests.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow})

# 3. Generate Hindi audio via XTTS (see Step 5 code)
```

This is a starting skeleton — building the full orchestrator (looping through every scene automatically) is a bigger project, but this is the right shape for it.

---

## VRAM BUDGET CHEAT SHEET FOR YOUR 24GB CARD

| Task | Approx VRAM needed | Fits on 7900 XTX? |
|---|---|---|
| SDXL image gen | ~8–10GB | Yes, comfortably |
| Flux Dev (FP8/GGUF quantized) | ~16–20GB | Yes, tight but fits |
| Flux Dev (BF16 full) | ~24GB | No — too tight, will OOM |
| WAN 2.2 video (quantized) | ~16–20GB | Yes |
| XTTS-v2 | ~4–6GB | Yes, easily, can run alongside image gen |
| Ollama 8B model | ~6–8GB | Yes |
| Ollama 14B model | ~10–12GB | Yes, but not simultaneously with a video gen job |

**Practical tip:** run image/video generation and the LLM as separate sequential steps rather than simultaneously — your 24GB is generous for one heavy job at a time, but running Ollama 14B + WAN 2.2 + XTTS all loaded at once will likely overflow. Your 64GB system RAM helps a lot here for offloading idle models to CPU between steps.

---

## RECOMMENDED ORDER OF OPERATIONS PER VIDEO

1. Run script generation (Ollama) — unload after
2. Run all character/environment image generation (ComfyUI + SDXL/Flux) — unload after
3. Run all video clip generation (ComfyUI + WAN 2.2/LTX) — unload after
4. Run all Hindi audio generation (XTTS) — lightweight, can run last
5. Assemble with ffmpeg (video + audio muxing) — CPU-only, no GPU needed

This sequential approach keeps you well within your 24GB VRAM ceiling at every step.

---
*This guide gets you a fully local version of the pipeline. Your 4 Bible files (Character, Acting, Voice, Environment) plug directly into the prompts at each step exactly as described in the Master Production Workflow.*
