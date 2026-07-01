from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path
import json
import sys

from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.bootstrap import _write_silent_reference_clip
from pipeline.common import load_voice_registry, save_voice_registry, timestamp_iso
from pipeline.media import synthesize_voice_clip


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "shared_character_catalog.json"
OUT_DIR = ROOT / "assets" / "shared_character_portraits"
VOICE_DIR = ROOT / "config" / "voices"

CHARACTER_STYLES = {
    "appy": {"bg": ((247, 238, 219), (255, 251, 243)), "mouth": "smile"},
    "nana": {"bg": ((255, 241, 171), (255, 248, 227)), "mouth": "open"},
    "ozzy": {"bg": ((224, 232, 244), (248, 250, 252)), "mouth": "flat"},
    "grapey": {"bg": ((226, 216, 255), (249, 244, 255)), "mouth": "smile"},
    "berry": {"bg": ((255, 214, 230), (255, 248, 250)), "mouth": "smile"},
    "mango": {"bg": ((255, 225, 176), (255, 248, 236)), "mouth": "smile"},
    "pinny": {"bg": ((255, 233, 181), (255, 251, 238)), "mouth": "smile"},
    "mellie": {"bg": ((208, 242, 211), (249, 255, 248)), "mouth": "smile"},
    "pappy": {"bg": ((247, 229, 202), (255, 249, 240)), "mouth": "smile"},
    "kiwi": {"bg": ((226, 212, 184), (249, 242, 231)), "mouth": "flat"},
    "coco": {"bg": ((221, 207, 197), (249, 243, 239)), "mouth": "flat"},
}


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    width, height = size
    img = Image.new("RGBA", size, top + (255,))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line((0, y, width, y), fill=color + (255,))
    return img


def _draw_common_face(draw: ImageDraw.ImageDraw, cx: int, cy: int, eye_color: tuple[int, int, int], mouth: str = "smile") -> None:
    draw.ellipse((cx - 44, cy - 34, cx - 22, cy - 12), fill=eye_color + (255,))
    draw.ellipse((cx + 22, cy - 34, cx + 44, cy - 12), fill=eye_color + (255,))
    draw.ellipse((cx - 37, cy - 28, cx - 29, cy - 20), fill=(255, 255, 255, 210))
    draw.ellipse((cx + 29, cy - 28, cx + 37, cy - 20), fill=(255, 255, 255, 210))
    if mouth == "smile":
        draw.arc((cx - 26, cy + 4, cx + 26, cy + 32), start=10, end=170, fill=(92, 52, 42, 255), width=5)
    elif mouth == "open":
        draw.ellipse((cx - 14, cy + 8, cx + 14, cy + 26), outline=(92, 52, 42, 255), width=4, fill=(245, 200, 188, 220))
    elif mouth == "flat":
        draw.line((cx - 20, cy + 18, cx + 20, cy + 18), fill=(92, 52, 42, 255), width=5)
    else:
        draw.arc((cx - 26, cy + 8, cx + 26, cy + 36), start=20, end=160, fill=(92, 52, 42, 255), width=4)


def _draw_leaf(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    draw.ellipse(box, fill=color + (255,), outline=(40, 90, 50, 255), width=3)


def _seed_for(entry: dict) -> int:
    text = f"{entry.get('character_id', '')}:{entry.get('name', '')}:{entry.get('fruit_type', '')}"
    return sum(ord(ch) for ch in text)


def _draw_accessories(draw: ImageDraw.ImageDraw, character_id: str, cx: int, cy: int) -> None:
    if character_id == "appy":
        draw.rounded_rectangle((265, 545, 395, 635), radius=18, fill=(118, 83, 43, 220), outline=(68, 46, 28, 255), width=4)
        draw.line((290, 580, 370, 580), fill=(245, 236, 219, 220), width=6)
    elif character_id == "nana":
        for dx, dy in [(-150, -170), (140, -150), (-120, 150), (130, 120)]:
            draw.polygon([(cx + dx, cy + dy - 12), (cx + dx + 9, cy + dy - 2), (cx + dx + 24, cy + dy), (cx + dx + 9, cy + dy + 10), (cx + dx + 3, cy + dy + 24), (cx + dx - 3, cy + dy + 10), (cx + dx - 18, cy + dy), (cx + dx - 3, cy + dy - 2)], fill=(255, 245, 158, 220))
    elif character_id == "ozzy":
        draw.rectangle((335, 260, 565, 300), fill=(28, 32, 43, 230))
        draw.rectangle((445, 300, 455, 360), fill=(30, 41, 59, 255))
        draw.polygon([(425, 360), (475, 360), (450, 420)], fill=(72, 38, 24, 220))
    elif character_id == "grapey":
        draw.ellipse((305, 520, 382, 598), outline=(88, 60, 24, 255), fill=(235, 204, 120, 220), width=4)
        draw.line((375, 540, 420, 500), fill=(88, 60, 24, 255), width=4)
        draw.ellipse((410, 492, 438, 520), fill=(255, 255, 255, 180), outline=(100, 84, 66, 255), width=2)
    elif character_id == "berry":
        draw.polygon([(460, 520), (500, 560), (460, 600), (420, 560)], fill=(255, 198, 214, 220), outline=(220, 110, 140, 255))
        draw.line((395, 560, 370, 585), fill=(220, 110, 140, 255), width=4)
        draw.line((495, 560, 520, 585), fill=(220, 110, 140, 255), width=4)
    elif character_id == "mango":
        draw.rounded_rectangle((300, 470, 600, 640), radius=36, outline=(120, 86, 54, 255), width=4, fill=(246, 228, 193, 190))
        draw.line((340, 505, 560, 505), fill=(120, 86, 54, 220), width=4)
        draw.arc((360, 485, 540, 620), start=20, end=160, fill=(120, 86, 54, 170), width=3)
    elif character_id == "pinny":
        draw.line((360, 530, 540, 530), fill=(87, 92, 94, 255), width=8)
        draw.line((450, 500, 450, 585), fill=(87, 92, 94, 255), width=8)
        draw.line((425, 505, 475, 555), fill=(87, 92, 94, 255), width=6)
    elif character_id == "mellie":
        draw.rounded_rectangle((320, 500, 580, 650), radius=34, fill=(255, 210, 222, 180), outline=(202, 126, 147, 160), width=3)
        draw.line((330, 540, 570, 540), fill=(202, 126, 147, 170), width=4)
        draw.polygon([(450, 470), (480, 500), (450, 530), (420, 500)], fill=(255, 170, 186, 220))
    elif character_id == "pappy":
        draw.line((560, 410, 590, 625), fill=(121, 88, 54, 255), width=9)
        draw.arc((525, 590, 615, 660), start=200, end=360, fill=(121, 88, 54, 255), width=7)
    elif character_id == "kiwi":
        draw.rounded_rectangle((310, 500, 590, 650), radius=40, outline=(108, 84, 55, 220), width=4, fill=(131, 158, 101, 170))
        draw.line((405, 535, 382, 585), fill=(108, 84, 55, 220), width=4)
        draw.line((495, 535, 518, 585), fill=(108, 84, 55, 220), width=4)
    elif character_id == "coco":
        draw.rectangle((340, 260, 560, 300), fill=(220, 223, 229, 220))
        draw.rectangle((446, 300, 454, 365), fill=(40, 44, 52, 255))
        draw.line((520, 530, 590, 590), fill=(72, 54, 36, 255), width=6)


def _write_voice_assets(entry: dict) -> tuple[Path, Path, str]:
    character_id = str(entry.get("character_id", ""))
    voice_id = str(entry.get("voice_id") or character_id)
    name = str(entry.get("name", character_id))
    personality = str(entry.get("personality", ""))
    acting = str(entry.get("acting", ""))
    seed = _seed_for(entry)
    ref_path = VOICE_DIR / f"{voice_id}_ref.wav"
    sample_path = VOICE_DIR / f"{voice_id}_sample.wav"
    _write_silent_reference_clip(ref_path, 155 + (seed % 9) * 18)
    sample_text = f"Hi, I am {name}. {personality}. {acting}."
    synthesize_voice_clip(
        sample_text,
        sample_path,
        reference_clip_path=ref_path,
        language="en",
        speed=0.92 + (seed % 4) * 0.08,
        voice_seed=seed,
    )
    return ref_path, sample_path, voice_id


def draw_character(entry: dict, out_path: Path) -> None:
    name = str(entry.get("name", "Character"))
    character_id = str(entry.get("character_id", ""))
    fruit = str(entry.get("fruit_type", "")).lower()
    palette = {
        "apple": ((252, 120, 133), (240, 247, 234), (238, 81, 96)),
        "banana": ((255, 238, 138), (254, 246, 195), (246, 209, 74)),
        "orange": ((255, 196, 104), (255, 240, 214), (250, 145, 35)),
        "grape": ((178, 140, 236), (244, 237, 255), (126, 82, 214)),
        "strawberry": ((255, 178, 191), (255, 243, 245), (234, 74, 105)),
        "mango": ((255, 184, 114), (255, 246, 217), (248, 143, 46)),
        "pineapple": ((255, 212, 117), (255, 250, 221), (217, 152, 37)),
        "watermelon": ((141, 220, 148), (236, 252, 237), (81, 174, 90)),
        "papaya": ((255, 186, 121), (255, 244, 227), (221, 115, 61)),
        "kiwi": ((185, 162, 120), (243, 236, 223), (116, 90, 56)),
        "coconut": ((180, 142, 106), (244, 232, 221), (95, 66, 47)),
    }
    style = CHARACTER_STYLES.get(character_id, {})
    bg_top, bg_bottom = style.get("bg", palette.get(fruit, ((220, 232, 255), (250, 251, 255), (100, 130, 200)))[:2])
    fruit_color = palette.get(fruit, ((220, 232, 255), (250, 251, 255), (100, 130, 200)))[2]
    img = _gradient((900, 900), bg_top, bg_bottom)
    draw = ImageDraw.Draw(img)
    # soft halo
    draw.ellipse((140, 110, 760, 730), fill=(255, 255, 255, 75))
    # floor shadow
    draw.ellipse((210, 700, 690, 820), fill=(80, 60, 50, 50))
    # body shape
    cx, cy = 450, 440
    body_bbox = (250, 170, 650, 650)
    if fruit == "banana":
        draw.rounded_rectangle((330, 170, 560, 650), radius=110, fill=fruit_color + (255,), outline=(92, 71, 32, 255), width=8)
        draw.line((385, 220, 350, 290), fill=(70, 50, 24, 255), width=8)
        draw.line((530, 220, 565, 290), fill=(70, 50, 24, 255), width=8)
        draw.ellipse((410, 140, 490, 205), fill=(40, 40, 40, 255))
        body_bbox = (330, 170, 560, 650)
    elif fruit == "grape":
        for ox, oy, r in [(-55, -55, 95), (45, -60, 95), (-10, 20, 95), (-80, 40, 90), (80, 40, 90), (0, 105, 88)]:
            draw.ellipse((cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r), fill=fruit_color + (255,), outline=(92, 71, 160, 255), width=8)
        draw.line((460, 120, 485, 70), fill=(70, 120, 70, 255), width=7)
        _draw_leaf(draw, (485, 50, 575, 110), (94, 185, 92))
        body_bbox = (285, 180, 620, 650)
    elif fruit == "strawberry":
        points = [(450, 160), (620, 300), (570, 640), (330, 640), (280, 300)]
        draw.polygon(points, fill=fruit_color + (255,), outline=(200, 54, 87, 255))
        for x in range(330, 570, 60):
            for y in range(290, 560, 70):
                draw.ellipse((x, y, x + 18, y + 18), fill=(255, 226, 110, 255))
        _draw_leaf(draw, (360, 90, 530, 165), (92, 178, 82))
        body_bbox = (280, 160, 620, 650)
    elif fruit == "watermelon":
        draw.ellipse(body_bbox, fill=fruit_color + (255,), outline=(49, 146, 63, 255), width=8)
        for x in range(300, 640, 52):
            draw.arc((x - 120, 175, x + 40, 650), start=270, end=90, fill=(100, 196, 113, 160), width=12)
        draw.ellipse((380, 120, 520, 200), fill=(92, 194, 95, 255))
        draw.line((450, 110, 460, 70), fill=(92, 194, 95, 255), width=8)
        body_bbox = (250, 170, 650, 650)
    elif fruit == "pineapple":
        draw.rounded_rectangle(body_bbox, radius=80, fill=(229, 170, 72, 255), outline=(150, 108, 28, 255), width=8)
        for y in range(210, 630, 42):
            draw.line((300, y, 600, y + 180), fill=(166, 122, 36, 255), width=4)
            draw.line((600, y, 300, y + 180), fill=(166, 122, 36, 255), width=4)
        for i in range(6):
            _draw_leaf(draw, (350 + i * 20, 40 - i * 10, 470 + i * 20, 200 - i * 10), (67, 165, 85))
        body_bbox = (260, 180, 640, 650)
    elif fruit == "kiwi":
        draw.ellipse(body_bbox, fill=fruit_color + (255,), outline=(104, 74, 44, 255), width=8)
        draw.ellipse((330, 260, 570, 500), fill=(111, 198, 90, 255), outline=(76, 130, 56, 255), width=5)
        for i in range(24):
            a = i * (2 * pi / 24)
            px = 450 + int(cos(a) * 60)
            py = 380 + int(sin(a) * 60)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(45, 42, 39, 255))
        draw.ellipse((410, 120, 490, 160), fill=(85, 55, 35, 255))
        body_bbox = (250, 170, 650, 650)
    elif fruit == "coconut":
        draw.ellipse(body_bbox, fill=fruit_color + (255,), outline=(86, 56, 38, 255), width=8)
        draw.arc((275, 195, 625, 625), start=20, end=160, fill=(120, 96, 74, 220), width=4)
        draw.arc((275, 195, 625, 625), start=200, end=340, fill=(120, 96, 74, 220), width=4)
        draw.ellipse((380, 120, 520, 175), fill=(104, 78, 54, 255))
        body_bbox = (250, 170, 650, 650)
    elif fruit == "apple":
        draw.ellipse(body_bbox, fill=fruit_color + (255,), outline=(145, 33, 46, 255), width=8)
        draw.ellipse((365, 100, 485, 170), fill=(84, 145, 64, 255))
        draw.line((460, 140, 470, 88), fill=(101, 72, 40, 255), width=7)
        body_bbox = (250, 170, 650, 650)
    elif fruit == "orange":
        draw.ellipse(body_bbox, fill=fruit_color + (255,), outline=(200, 113, 28, 255), width=8)
        for r in range(140, 200, 8):
            draw.arc((320, 220, 580, 580), start=0, end=360, fill=(255, 220, 158, 28), width=2)
        draw.ellipse((390, 100, 490, 165), fill=(90, 149, 61, 255))
        body_bbox = (250, 170, 650, 650)
    elif fruit == "mango":
        draw.ellipse((265, 210, 620, 650), fill=fruit_color + (255,), outline=(205, 100, 46, 255), width=8)
        draw.ellipse((375, 110, 490, 185), fill=(85, 153, 70, 255))
        draw.line((450, 150, 465, 96), fill=(90, 75, 45, 255), width=7)
        body_bbox = (265, 210, 620, 650)
    elif fruit == "papaya":
        draw.ellipse((275, 200, 625, 650), fill=fruit_color + (255,), outline=(176, 94, 57, 255), width=8)
        draw.ellipse((355, 105, 505, 190), fill=(94, 160, 80, 255))
        draw.line((450, 150, 452, 92), fill=(90, 75, 45, 255), width=7)
        body_bbox = (275, 200, 625, 650)

    # outfit accents
    if fruit in {"apple", "orange", "mango", "papaya"}:
        draw.rounded_rectangle((315, 485, 585, 650), radius=40, fill=(82, 56, 35, 180))
        draw.line((340, 520, 560, 520), fill=(250, 241, 224, 200), width=7)
    elif fruit == "banana":
        draw.polygon([(340, 470), (560, 470), (530, 650), (360, 650)], fill=(255, 226, 120, 185))
        draw.line((355, 520, 545, 520), fill=(255, 255, 255, 180), width=6)
    elif fruit == "watermelon":
        draw.rounded_rectangle((330, 500, 570, 650), radius=40, fill=(255, 180, 208, 180))
    elif fruit == "coconut":
        draw.rounded_rectangle((320, 500, 580, 650), radius=40, fill=(255, 255, 255, 185))
        draw.line((345, 540, 555, 540), fill=(220, 220, 220, 220), width=4)
    elif fruit == "kiwi":
        draw.rounded_rectangle((330, 500, 570, 650), radius=40, fill=(120, 165, 95, 180))
    elif fruit == "pineapple":
        draw.rounded_rectangle((320, 500, 580, 650), radius=40, fill=(246, 231, 193, 185))
    elif fruit == "grape":
        draw.rounded_rectangle((330, 505, 570, 650), radius=40, fill=(90, 55, 120, 170))
    elif fruit == "strawberry":
        draw.rounded_rectangle((325, 500, 575, 650), radius=40, fill=(255, 214, 231, 185))
    else:
        draw.rounded_rectangle((325, 500, 575, 650), radius=40, fill=(255, 255, 255, 170))

    # face
    mouth = str(style.get("mouth", "smile"))
    _draw_common_face(draw, cx, 360 if fruit == "banana" else 370, (58, 44, 38), mouth if mouth else ("smile" if fruit not in {"kiwi", "coconut"} else "flat"))
    if fruit == "banana":
        draw.arc((400, 392, 500, 450), start=20, end=160, fill=(92, 52, 42, 255), width=5)
    if fruit == "kiwi":
        draw.ellipse((410, 280, 438, 308), fill=(45, 32, 28, 255))
        draw.ellipse((462, 280, 490, 308), fill=(45, 32, 28, 255))
        draw.line((430, 332, 470, 332), fill=(92, 52, 42, 255), width=4)
    if fruit == "coconut":
        draw.ellipse((404, 285, 438, 319), fill=(33, 33, 36, 255))
        draw.ellipse((462, 285, 496, 319), fill=(33, 33, 36, 255))
        draw.line((430, 334, 472, 334), fill=(92, 52, 42, 255), width=4)

    _draw_accessories(draw, character_id, cx, cy)

    # title label
    title_font = _font(44)
    subtitle_font = _font(24)
    draw.rounded_rectangle((58, 40, 842, 108), radius=26, fill=(255, 255, 255, 150))
    draw.text((78, 50), name, fill=(30, 41, 59, 255), font=title_font)
    draw.text((79, 92), str(entry.get("file_name", "")), fill=(71, 85, 105, 255), font=subtitle_font)
    draw.rounded_rectangle((110, 760, 790, 835), radius=25, fill=(255, 255, 255, 150))
    draw.text((128, 772), str(entry.get("fruit_type", "")), fill=(30, 41, 59, 255), font=_font(34))
    draw.text((450, 772), "Shared library character", fill=(71, 85, 105, 255), font=_font(28))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    entries = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    voice_registry = load_voice_registry()
    for entry in entries:
        character_id = str(entry["character_id"])
        out_path = OUT_DIR / f"{character_id}.png"
        draw_character(entry, out_path)
        ref_path, sample_path, voice_id = _write_voice_assets(entry)
        voice_registry[voice_id] = {
            "reference_clip": f"config/voices/{ref_path.name}",
            "sample_clip": f"config/voices/{sample_path.name}",
            "language": "en",
            "pitch_shift": 0,
            "speed": round(0.92 + (_seed_for(entry) % 4) * 0.08, 2),
            "created": timestamp_iso(),
        }
        print(out_path)
        print(sample_path)
    save_voice_registry(voice_registry)


if __name__ == "__main__":
    main()
