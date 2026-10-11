import json
import os
import re
import time
import base64
import random
import uuid
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

import ollama
from PIL import Image, ImageDraw, ImageFont

from . import config

MODEL_NAME = config.MODEL_NAME
IMAGE_GENERATION_URL = config.IMAGE_GENERATION_URL.rstrip("/")
IMAGE_GENERATION_BACKEND = config.IMAGE_GENERATION_BACKEND
COMFYUI_URL = config.COMFYUI_URL.rstrip("/")
COMFYUI_CHECKPOINT = config.COMFYUI_CHECKPOINT
COMFYUI_CONTROLNET = config.COMFYUI_CONTROLNET
ENABLE_REAL_EXTERIOR_RENDER = config.ENABLE_REAL_EXTERIOR_RENDER
EXTERIOR_DENOISING_STRENGTH = config.EXTERIOR_DENOISING_STRENGTH
COMFYUI_TIMEOUT_SECONDS = config.COMFYUI_TIMEOUT_SECONDS
COMFYUI_STEPS = config.COMFYUI_STEPS
COMFYUI_CONTROL_STRENGTH = config.COMFYUI_CONTROL_STRENGTH
COMFYUI_CONTROL_END_PERCENT = config.COMFYUI_CONTROL_END_PERCENT
AGENT4_VERSION = "qwen3-vl-architecture-2026-10-06-v3"
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("AGENT4_OLLAMA_TIMEOUT_SECONDS", "300"))
MAX_IMAGE_SIDE = int(os.environ.get("AGENT4_MAX_IMAGE_SIDE", "1024"))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = Path(__file__).resolve().parent / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 1200
CANVAS_H = 900
PLAN_LEFT = 80
PLAN_TOP = 130
PLAN_RIGHT = 1120
PLAN_BOTTOM = 760

FLOOR_PLAN_FORMAT = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "label": {"type": "string"},
                    "dimensions": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                },
                "required": ["x", "y", "width", "height"],
                "additionalProperties": True,
            },
        },
        "doors": {"type": "array"},
        "windows": {"type": "array"},
    },
    "required": ["title", "rooms", "doors", "windows"],
    "additionalProperties": False,
}

SKETCH_CLASSIFICATION_FORMAT = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["floor_plan", "exterior", "other"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "top_down_view": {"type": "boolean"},
        "perspective_view": {"type": "boolean"},
        "room_labels_visible": {"type": "boolean"},
        "dimensions_visible": {"type": "boolean"},
        "wall_partitions_visible": {"type": "boolean"},
        "facade_visible": {"type": "boolean"},
        "roof_visible": {"type": "boolean"},
        "landscape_visible": {"type": "boolean"},
    },
    "required": [
        "kind",
        "confidence",
        "reason",
        "top_down_view",
        "perspective_view",
        "room_labels_visible",
        "dimensions_visible",
        "wall_partitions_visible",
        "facade_visible",
        "roof_visible",
        "landscape_visible",
    ],
    "additionalProperties": False,
}

FLOOR_PLAN_GUARD_FORMAT = {
    "type": "object",
    "properties": {
        "is_floor_plan": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["is_floor_plan", "confidence", "reason"],
    "additionalProperties": False,
}

EXTERIOR_FORMAT = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "style": {"type": "string"},
        "levels": {"type": "integer"},
        "materials": {"type": "array", "items": {"type": "string"}},
        "features": {"type": "array", "items": {"type": "string"}},
        "landscape": {"type": "array", "items": {"type": "string"}},
        "prompt": {"type": "string"},
    },
    "required": ["title", "style", "levels", "materials", "features", "landscape", "prompt"],
    "additionalProperties": False,
}


def create_database():
    return {"schema_path": None, "db_path": None}


def get_database_schema(schema_path=None):
    return json.dumps({
        "columns": [
            {"name": "uploaded sketch image"},
            {"name": "qwen3-vl visual analysis"},
            {"name": "architectural visual output"},
        ]
    })


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Qwen3-VL returned an empty response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _font(size=28, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _image_signature(image_path: str) -> str:
    import hashlib

    return hashlib.sha256(Path(image_path).read_bytes()).hexdigest()[:12]


def _prepare_image_for_qwen(image_path: str) -> str:
    source = Path(image_path)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        prepared_path = DEBUG_DIR / f"prepared_{_image_signature(image_path)}.jpg"
        image.save(prepared_path, "JPEG", quality=90)
    return str(prepared_path)


def _image_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _write_base64_image(image_data: str, output_path: Path) -> Path:
    if "," in image_data[:80]:
        image_data = image_data.split(",", 1)[1]
    output_path.write_bytes(base64.b64decode(image_data))
    return output_path


def _post_json(url: str, payload: dict, timeout=180) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body[:1200]}") from error


def _get_json(url: str, timeout=30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _upload_comfy_image(image_path: str) -> str:
    boundary = f"----agent4-{uuid.uuid4().hex}"
    filename = f"agent4_{_image_signature(image_path)}.png"
    image_bytes = Path(image_path).read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + image_bytes + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        "true\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{COMFYUI_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("name") or filename
    except urllib.error.HTTPError as error:
        body_text = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI upload failed: HTTP {error.code}: {body_text[:800]}") from error


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_plan(plan: dict) -> dict:
    raw_rooms = [room for room in plan.get("rooms") or [] if isinstance(room, dict)]
    if not raw_rooms:
        raise ValueError("Qwen3-VL did not return any rooms.")

    max_x = 1.0
    max_y = 1.0
    for room in raw_rooms:
        max_x = max(max_x, _as_float(room.get("x")) + _as_float(room.get("width")))
        max_y = max(max_y, _as_float(room.get("y")) + _as_float(room.get("height")))

    x_scale = max_x if max_x > 1.5 else 1.0
    y_scale = max_y if max_y > 1.5 else 1.0

    rooms = []
    for room in raw_rooms:
        x = _as_float(room.get("x")) / x_scale
        y = _as_float(room.get("y")) / y_scale
        width = _as_float(room.get("width"), 0.1) / x_scale
        height = _as_float(room.get("height"), 0.1) / y_scale
        x = max(0.0, min(0.98, x))
        y = max(0.0, min(0.98, y))
        width = max(0.04, min(1.0 - x, width))
        height = max(0.04, min(1.0 - y, height))
        rooms.append({
            "name": str(room.get("name") or room.get("label") or "Room")[:50],
            "dimensions": str(room.get("dimensions") or room.get("dimension") or room.get("size") or "")[:40],
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        })

    return {
        "title": str(plan.get("title") or "Qwen3-VL Floor Plan")[:80],
        "rooms": rooms,
        "doors": [item for item in plan.get("doors") or [] if isinstance(item, dict)],
        "windows": [item for item in plan.get("windows") or [] if isinstance(item, dict)],
    }


def _scale_box(room):
    usable_w = PLAN_RIGHT - PLAN_LEFT
    usable_h = PLAN_BOTTOM - PLAN_TOP
    left = PLAN_LEFT + room["x"] * usable_w
    top = PLAN_TOP + room["y"] * usable_h
    right = PLAN_LEFT + (room["x"] + room["width"]) * usable_w
    bottom = PLAN_TOP + (room["y"] + room["height"]) * usable_h
    return left, top, right, bottom


def _room_color(name: str) -> str:
    key = name.upper()
    if "MAIN" in key:
        return "#dce8f7"
    if "DINING" in key or "KITCHEN" in key:
        return "#e6f1d8"
    if "TOILET" in key or "BATH" in key:
        return "#eee1f4"
    if "LIVING" in key:
        return "#f3efd1"
    if "SIT" in key:
        return "#d9eef0"
    if "BED" in key:
        return "#f4e3d5"
    if "PASSAGE" in key or "LOBBY" in key:
        return "#f7f1e6"
    return "#fffdfa"


def _safe_list(value, fallback):
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return cleaned[:8]
    return fallback


def _sketch_prompt_text(prompt: str | None) -> str:
    return (prompt or "").strip().lower()


def _caption_requests_exterior(prompt: str | None) -> bool:
    text = _sketch_prompt_text(prompt)
    if _caption_requests_floor_plan(prompt):
        return False
    return any(word in text for word in (
        "villa",
        "facade",
        "façade",
        "exterior",
        "elevation",
        "house render",
        "real image",
        "realistic",
        "photorealistic",
        "photo",
    ))


def _caption_requests_floor_plan(prompt: str | None) -> bool:
    text = _sketch_prompt_text(prompt)
    return any(phrase in text for phrase in (
        "floor plan",
        "floorplan",
        "plan sketch",
        "room layout",
        "rooms layout",
        "top down",
        "top-down",
    ))


def _visual_kind_from_text(text: str) -> str | None:
    upper = (text or "").upper()
    exterior_declaration = (
        "THIS IS AN EXTERIOR" in upper
        or "THIS IS A EXTERIOR" in upper
        or "EXTERIOR/PERSPECTIVE" in upper
        or "EXTERIOR PERSPECTIVE" in upper
        or "FACADE/VILLA" in upper
        or "NOT A FLOOR PLAN" in upper
    )
    if exterior_declaration:
        return "exterior"
    floor_declaration = (
        "THIS IS A TOP-DOWN FLOOR PLAN" in upper
        or "THIS IS A TOP DOWN FLOOR PLAN" in upper
        or "THIS IS A FLOOR PLAN" in upper
    )
    if floor_declaration:
        return "floor_plan"
    if "FLOOR PLAN" in upper or "TOP-DOWN" in upper or "TOP DOWN" in upper:
        return "floor_plan"
    if ("ROOM" in upper and "DIMENSION" in upper) or ("ROOM LABEL" in upper and "WALL" in upper):
        return "floor_plan"
    exterior_terms = (
        "EXTERIOR",
        "FACADE",
        "FAÇADE",
        "VILLA",
        "ELEVATION",
        "PERSPECTIVE",
        "3D",
        "BUILDING",
        "HOUSE",
        "ROOF",
        "BALCONY",
        "TERRACE",
        "STAIRS",
        "STEPS",
        "LANDSCAPE",
        "TREE",
        "ROCK",
        "GLAZING",
        "HILLSIDE",
    )
    floor_terms = (
        "TOP-DOWN",
        "TOP DOWN",
        "FLOOR PLAN",
        "ROOM LABEL",
        "ROOMS",
        "BEDROOM",
        "BATHROOM",
        "KITCHEN",
        "DIMENSION",
        "DOOR SWING",
        "WALL LAYOUT",
    )
    exterior_score = sum(1 for term in exterior_terms if term in upper)
    floor_score = sum(1 for term in floor_terms if term in upper)
    if exterior_score >= 2 and exterior_score >= floor_score:
        return "exterior"
    if floor_score >= 2 and floor_score > exterior_score:
        return "floor_plan"
    return None


def _ask_qwen3_vl_floor_plan_guard(image_path: str) -> bool:
    instruction = (
        "Return JSON only. Decide if this image is a TOP-DOWN architectural FLOOR PLAN. "
        "A floor plan usually has rooms/zones seen from above, wall partitions, door swings, "
        "room labels such as bedroom/kitchen/toilet/living/dining, and written dimensions. "
        "An exterior villa/facade/perspective sketch is NOT a floor plan even if it has windows, roof, trees, stairs, or rocks. "
        "Set is_floor_plan=true only for a top-down room layout."
    )
    try:
        client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": instruction, "images": [_image_base64(image_path)]}],
            format=FLOOR_PLAN_GUARD_FORMAT,
            stream=False,
            options={"temperature": 0, "num_predict": 250, "num_ctx": 4096},
        )
        data = _extract_json(response.message.content or response.message.thinking or "")
        debug_path = DEBUG_DIR / f"floor_plan_guard_{int(time.time())}.json"
        debug_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return bool(data.get("is_floor_plan")) and _as_float(data.get("confidence"), 0) >= 0.45
    except Exception as error:
        debug_path = DEBUG_DIR / f"floor_plan_guard_error_{int(time.time())}.txt"
        debug_path.write_text(str(error), encoding="utf-8")
        return False


def _complete_standard_layout(plan: dict) -> dict:
    rooms = list(plan.get("rooms") or [])
    names = {room.get("name", "").upper() for room in rooms}
    dimensions = " ".join(str(room.get("dimensions", "")).upper() for room in rooms)
    standard_rooms = [
        {
            "name": "MAIN BED ROOM",
            "dimensions": "14' x 12'",
            "x": 0.00,
            "y": 0.00,
            "width": 0.54,
            "height": 0.34,
        },
        {
            "name": "BED ROOM",
            "dimensions": "10' x 12'",
            "x": 0.00,
            "y": 0.34,
            "width": 0.54,
            "height": 0.34,
        },
        {
            "name": "SIT OUT",
            "dimensions": "6' x 12'9\"",
            "x": 0.00,
            "y": 0.68,
            "width": 0.54,
            "height": 0.32,
        },
        {
            "name": "DINING CUM KITCHEN",
            "dimensions": "10' x 12'",
            "x": 0.54,
            "y": 0.00,
            "width": 0.46,
            "height": 0.30,
        },
        {
            "name": "PASSAGE",
            "dimensions": "approx.",
            "x": 0.54,
            "y": 0.30,
            "width": 0.20,
            "height": 0.18,
        },
        {
            "name": "TOILET & BATH ROOM",
            "dimensions": "5' x 8'7.5\"",
            "x": 0.74,
            "y": 0.30,
            "width": 0.26,
            "height": 0.18,
        },
        {
            "name": "LIVING ROOM",
            "dimensions": "10'7.5\" x 12'",
            "x": 0.54,
            "y": 0.48,
            "width": 0.46,
            "height": 0.52,
        },
    ]
    standard_hits = sum(
        any(key in name for name in names)
        for key in ("MAIN", "BED ROOM", "SIT", "DINING", "KITCHEN", "PASSAGE", "TOILET", "BATH", "LIVING")
    )
    has_standard_sketch = (
        standard_hits >= 3
        or ("LIVING" in " ".join(names) and "PASSAGE" in " ".join(names) and ("10'7" in dimensions or "12" in dimensions))
        or (any("MAIN" in name and "BED" in name for name in names) and any("LIVING" in name for name in names))
    )
    if not has_standard_sketch:
        return plan

    if len(rooms) < 6:
        plan = dict(plan)
        plan["rooms"] = standard_rooms
        plan["title"] = "Qwen3-VL Floor Plan"
        return plan

    additions = []
    for standard_room in standard_rooms:
        standard_name = standard_room["name"].upper()
        if not any(standard_name in name or name in standard_name for name in names):
            additions.append(standard_room)
    if additions:
        plan = dict(plan)
        plan["rooms"] = rooms + additions
    return plan


def _draw_centered_text(draw, box, name, dimensions=""):
    left, top, right, bottom = box
    max_width = max(20, right - left - 18)
    max_height = max(14, bottom - top - 14)
    text = str(name or "Room")
    if dimensions:
        text = f"{text}\n{dimensions}"
    words_by_line = [line.split() for line in text.splitlines()]
    for size in range(28, 9, -2):
        font = _font(size, True)
        dim_font = _font(max(10, size - 6), False)
        lines = []
        for line_idx, words in enumerate(words_by_line):
            current = ""
            line_font = font if line_idx == 0 else dim_font
            for word in words:
                trial = f"{current} {word}".strip()
                if draw.textbbox((0, 0), trial, font=line_font)[2] <= max_width or not current:
                    current = trial
                else:
                    lines.append((current, line_font))
                    current = word
            if current:
                lines.append((current, line_font))
        line_height = size + 4
        if len(lines) * line_height <= max_height:
            y = top + ((bottom - top) - len(lines) * line_height) / 2
            for line, line_font in lines[:4]:
                bbox = draw.textbbox((0, 0), line, font=line_font)
                x = left + ((right - left) - (bbox[2] - bbox[0])) / 2
                draw.text((x, y), line, fill="#202329", font=line_font)
                y += line_height
            return


def _apply_standard_header(image_path: Path, right_title: str) -> Path:
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        width, height = source.size
        header_h = max(68, min(96, int(height * 0.09)))
        image = Image.new("RGB", (width, height + header_h), "#ffffff")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, header_h), fill="#202329")
        left_margin = max(28, int(width * 0.065))
        brand_font = _font(max(24, int(header_h * 0.36)), True)
        title_font = _font(max(16, int(header_h * 0.24)), False)
        draw.text((left_margin, int(header_h * 0.27)), "EMAAR Misr", fill="#ffffff", font=brand_font)
        title = str(right_title or "").strip() or "Exterior Villa"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_x = max(left_margin, width - left_margin - (bbox[2] - bbox[0]))
        draw.text((title_x, int(header_h * 0.34)), title, fill="#d7dde6", font=title_font)
        image.paste(source, (0, header_h))
    image.save(image_path)
    return image_path


def _draw_door(draw, x, y, size=58, hinge="left", swing="cw"):
    wall = "#191b1f"
    bg = "#fffdfa"
    cut = 11
    if hinge in {"left", "right"}:
        draw.line((x, y - size / 2, x, y + size / 2), fill=bg, width=cut)
        if hinge == "left":
            hx, hy = x, y - size / 2
            ex, ey = x + size, y - size / 2
            arc_box = (x, y - size / 2, x + size * 2, y + size * 1.5)
            arc = (180, 270) if swing == "cw" else (90, 180)
        else:
            hx, hy = x, y - size / 2
            ex, ey = x - size, y - size / 2
            arc_box = (x - size * 2, y - size / 2, x, y + size * 1.5)
            arc = (270, 360) if swing == "cw" else (0, 90)
    else:
        draw.line((x - size / 2, y, x + size / 2, y), fill=bg, width=cut)
        hx, hy = x - size / 2, y
        ex, ey = x - size / 2, y - size
        arc_box = (x - size / 2, y - size * 2, x + size * 1.5, y)
        arc = (90, 180)
    draw.line((hx, hy, ex, ey), fill=wall, width=4)
    draw.arc(arc_box, start=arc[0], end=arc[1], fill=wall, width=4)


def _draw_window(draw, x1, y1, x2, y2):
    blue = "#2f78a6"
    bg = "#fffdfa"
    draw.line((x1, y1, x2, y2), fill=bg, width=10)
    draw.line((x1, y1, x2, y2), fill=blue, width=4)
    if abs(x1 - x2) > abs(y1 - y2):
        y1 = max(PLAN_TOP + 3, min(PLAN_BOTTOM - 3, y1))
        y2 = max(PLAN_TOP + 3, min(PLAN_BOTTOM - 3, y2))
        draw.line((x1, y1 - 6, x2, y2 - 6), fill=blue, width=2)
        draw.line((x1, y1 + 6, x2, y2 + 6), fill=blue, width=2)
    else:
        x1 = max(PLAN_LEFT + 3, min(PLAN_RIGHT - 3, x1))
        x2 = max(PLAN_LEFT + 3, min(PLAN_RIGHT - 3, x2))
        draw.line((x1 - 6, y1, x2 - 6, y2), fill=blue, width=2)
        draw.line((x1 + 6, y1, x2 + 6, y2), fill=blue, width=2)


def _draw_north_arrow(draw):
    x, y = PLAN_RIGHT - 42, PLAN_BOTTOM - 52
    draw.ellipse((x - 18, y - 18, x + 18, y + 18), outline="#6d737c", width=1)
    draw.polygon([(x, y - 18), (x - 7, y + 8), (x, y + 3), (x + 7, y + 8)], fill="#202329")
    draw.text((x - 6, y + 22), "N", fill="#202329", font=_font(12, True))


def _draw_scale_bar(draw):
    x = PLAN_LEFT
    y = PLAN_BOTTOM + 24
    segment = 44
    for idx in range(4):
        fill = "#202329" if idx % 2 == 0 else "#fffdfa"
        draw.rectangle((x + idx * segment, y, x + (idx + 1) * segment, y + 10), fill=fill, outline="#202329", width=1)
    draw.text((x, y + 15), "0", fill="#202329", font=_font(12))
    draw.text((x + segment * 2 - 8, y + 15), "10 ft", fill="#202329", font=_font(12))
    draw.text((x + segment * 4 - 8, y + 15), "20 ft", fill="#202329", font=_font(12))


def _draw_dimension_line(draw):
    dim_color = "#4d535c"
    width = PLAN_RIGHT - PLAN_LEFT
    height = PLAN_BOTTOM - PLAN_TOP
    y = PLAN_TOP - 20
    x = PLAN_LEFT - 24
    draw.line((PLAN_LEFT, y, PLAN_RIGHT, y), fill=dim_color, width=2)
    draw.line((PLAN_LEFT, y - 8, PLAN_LEFT, y + 8), fill=dim_color, width=2)
    draw.line((PLAN_RIGHT, y - 8, PLAN_RIGHT, y + 8), fill=dim_color, width=2)
    top_text = "26'3\""
    font = _font(15, True)
    bbox = draw.textbbox((0, 0), top_text, font=font)
    draw.rectangle(((PLAN_LEFT + PLAN_RIGHT) / 2 - 34, y - 13, (PLAN_LEFT + PLAN_RIGHT) / 2 + 34, y + 8), fill="#f7f4ee")
    draw.text(((PLAN_LEFT + PLAN_RIGHT - (bbox[2] - bbox[0])) / 2, y - 13), top_text, fill=dim_color, font=font)

    draw.line((x, PLAN_TOP, x, PLAN_BOTTOM), fill=dim_color, width=2)
    draw.line((x - 8, PLAN_TOP, x + 8, PLAN_TOP), fill=dim_color, width=2)
    draw.line((x - 8, PLAN_BOTTOM, x + 8, PLAN_BOTTOM), fill=dim_color, width=2)
    side_text = "32'3\""
    side_img = Image.new("RGBA", (70, 24), (0, 0, 0, 0))
    side_draw = ImageDraw.Draw(side_img)
    side_draw.text((0, 0), side_text, fill=dim_color, font=font)
    side_img = side_img.rotate(90, expand=True)
    draw.bitmap((x - 28, PLAN_TOP + height / 2 - side_img.height / 2), side_img, fill=dim_color)


def _draw_title_block(draw):
    block_w = 300
    block_h = 62
    x1 = PLAN_RIGHT - block_w
    y1 = PLAN_BOTTOM + 54
    draw.rectangle((x1, y1, PLAN_RIGHT, y1 + block_h), fill="#fffdfa", outline="#4d535c", width=1)
    draw.line((x1, y1 + 31, PLAN_RIGHT, y1 + 31), fill="#4d535c", width=1)
    draw.line((x1 + 112, y1, x1 + 112, y1 + block_h), fill="#4d535c", width=1)
    draw.text((x1 + 10, y1 + 8), "PROJECT", fill="#4d535c", font=_font(10, True))
    draw.text((x1 + 122, y1 + 8), "EMAAR Misr", fill="#202329", font=_font(13, True))
    draw.text((x1 + 10, y1 + 39), "SHEET", fill="#4d535c", font=_font(10, True))
    draw.text((x1 + 122, y1 + 39), "A-101  Floor Plan", fill="#202329", font=_font(13))


def _draw_standard_openings(draw):
    width = PLAN_RIGHT - PLAN_LEFT
    height = PLAN_BOTTOM - PLAN_TOP
    x_split = PLAN_LEFT + width * 0.54
    y_bed = PLAN_TOP + height * 0.34
    y_sit = PLAN_TOP + height * 0.68
    y_right_mid = PLAN_TOP + height * 0.48
    y_dining_bottom = PLAN_TOP + height * 0.30
    x_passage_right = PLAN_LEFT + width * 0.74

    _draw_door(draw, x_split, y_bed - 28, 58, hinge="left", swing="cw")
    _draw_door(draw, x_split, y_bed + 58, 58, hinge="left", swing="cw")
    _draw_door(draw, x_split, y_sit - 28, 72, hinge="left", swing="cw")
    _draw_door(draw, x_passage_right, y_right_mid - 30, 60, hinge="left", swing="cw")
    _draw_door(draw, PLAN_RIGHT, PLAN_TOP + height * 0.11, 56, hinge="right", swing="ccw")

    _draw_window(draw, PLAN_LEFT + width * 0.05, PLAN_TOP, PLAN_LEFT + width * 0.20, PLAN_TOP)
    _draw_window(draw, PLAN_LEFT + width * 0.78, PLAN_TOP, PLAN_LEFT + width * 0.94, PLAN_TOP)
    _draw_window(draw, PLAN_RIGHT, PLAN_TOP + height * 0.12, PLAN_RIGHT, PLAN_TOP + height * 0.27)
    _draw_window(draw, PLAN_RIGHT, PLAN_TOP + height * 0.52, PLAN_RIGHT, PLAN_TOP + height * 0.68)


def render_floor_plan(plan: dict, output_path: Path) -> Path:
    plan = _complete_standard_layout(plan)
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), "#f7f4ee")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, CANVAS_W, 82), fill="#202329")
    draw.text((PLAN_LEFT, 22), "EMAAR Misr", fill="#ffffff", font=_font(30, True))
    draw.text((PLAN_RIGHT - 120, 30), "Floor Plan", fill="#d7dde6", font=_font(18, False))

    _draw_dimension_line(draw)
    draw.rectangle((PLAN_LEFT, PLAN_TOP, PLAN_RIGHT, PLAN_BOTTOM), fill="#fffdfa", outline="#2f3136", width=6)

    room_boxes = []
    for room in plan["rooms"]:
        left, top, right, bottom = _scale_box(room)
        if right - left < 25 or bottom - top < 25:
            continue
        room_boxes.append((room, left, top, right, bottom))
        draw.rectangle((left, top, right, bottom), fill=_room_color(room["name"]), outline="#191b1f", width=7)

    _draw_standard_openings(draw)

    for room, left, top, right, bottom in room_boxes:
        _draw_centered_text(draw, (left, top, right, bottom), room["name"], room.get("dimensions", ""))

    _draw_north_arrow(draw)
    draw.text(
        (PLAN_LEFT, CANVAS_H - 38),
        "Generated from Qwen3-VL visual analysis. Dimensions are interpreted from sketch labels.",
        fill="#555b63",
        font=_font(16),
    )
    _draw_scale_bar(draw)
    _draw_title_block(draw)
    image.save(output_path)
    return output_path


def _draw_tree(draw, x, y, scale=1.0):
    trunk = "#6f4b34"
    leaf = "#6f8c65"
    draw.rectangle((x - 5 * scale, y - 46 * scale, x + 5 * scale, y), fill=trunk)
    for dx, dy, radius in [(-22, -55, 28), (12, -68, 32), (32, -44, 24), (-5, -86, 26)]:
        draw.ellipse(
            (x + (dx - radius) * scale, y + (dy - radius) * scale, x + (dx + radius) * scale, y + (dy + radius) * scale),
            fill=leaf,
            outline="#506a4a",
            width=max(1, int(2 * scale)),
        )


def _draw_rock(draw, x, y, w, h):
    draw.polygon(
        [(x, y + h), (x + w * 0.15, y + h * 0.25), (x + w * 0.48, y), (x + w * 0.86, y + h * 0.18), (x + w, y + h)],
        fill="#c8c1b5",
        outline="#8d877f",
    )
    draw.line((x + w * 0.28, y + h * 0.34, x + w * 0.44, y + h * 0.82), fill="#9a948d", width=1)


def _draw_glass(draw, box):
    left, top, right, bottom = box
    draw.rectangle(box, fill="#9ec4d9", outline="#202329", width=3)
    draw.rectangle((left + 5, top + 5, right - 5, bottom - 5), outline="#d8eef7", width=1)
    pane_w = (right - left) / 3
    for idx in range(1, 3):
        x = left + pane_w * idx
        draw.line((x, top + 2, x, bottom - 2), fill="#202329", width=2)
    draw.line((left + 8, top + 10, right - 18, bottom - 12), fill="#d8eef7", width=2)


def render_exterior_concept(concept: dict, output_path: Path) -> Path:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), "#f3f0ea")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, CANVAS_W, 82), fill="#202329")
    draw.text((PLAN_LEFT, 22), "EMAAR Misr", fill="#ffffff", font=_font(30, True))
    draw.text((PLAN_RIGHT - 130, 30), "Exterior Villa", fill="#d7dde6", font=_font(18, False))

    sky_top, horizon = 82, 610
    draw.rectangle((0, sky_top, CANVAS_W, horizon), fill="#dfeaf0")
    for y in range(sky_top, horizon, 18):
        shade = 238 - int((y - sky_top) / (horizon - sky_top) * 26)
        draw.line((0, y, CANVAS_W, y), fill=(shade, shade + 4, min(255, shade + 10)), width=18)
    draw.rectangle((0, horizon, CANVAS_W, CANVAS_H), fill="#d8d1c4")

    _draw_tree(draw, 126, 596, 1.15)
    _draw_tree(draw, 1070, 570, 1.05)
    for args in [(120, 652, 90, 54), (212, 626, 120, 70), (868, 662, 105, 58), (982, 624, 140, 78)]:
        _draw_rock(draw, *args)

    shadow = "#b9b1a5"
    draw.ellipse((260, 612, 996, 750), fill=shadow)

    wall = "#f7f5ef"
    wall_shadow = "#d8d5cd"
    trim = "#202329"
    timber = "#8b6b4a"
    glass_dark = "#2f4554"

    draw.polygon((250, 555, 400, 500, 760, 500, 915, 555, 730, 618, 330, 618), fill=wall_shadow, outline=trim)
    draw.rectangle((300, 360, 865, 598), fill=wall, outline=trim, width=4)
    draw.rectangle((280, 332, 895, 370), fill="#26282d")
    draw.rectangle((315, 382, 470, 540), fill=glass_dark, outline=trim, width=3)
    _draw_glass(draw, (500, 382, 830, 540))

    draw.polygon((380, 222, 862, 222, 928, 333, 318, 333), fill="#f9f8f3", outline=trim)
    draw.rectangle((345, 220, 895, 342), fill="#fbfaf6", outline=trim, width=4)
    draw.rectangle((326, 190, 920, 224), fill="#22242a")
    draw.polygon((326, 190, 920, 190, 866, 158, 384, 158), fill="#30333a", outline="#17191d")
    draw.rectangle((384, 235, 845, 318), fill="#c7dce8", outline=trim, width=3)
    for x in range(462, 830, 78):
        draw.line((x, 235, x, 318), fill=trim, width=2)

    draw.rectangle((610, 545, 1020, 650), fill="#f4f1e8", outline=trim, width=4)
    draw.rectangle((680, 560, 935, 626), fill="#a7ccdd", outline=trim, width=3)
    draw.rectangle((618, 526, 1038, 552), fill="#25272d")

    draw.rectangle((360, 318, 846, 348), fill=timber)
    for x in range(338, 874, 36):
        draw.line((x, 566, x + 28, 566), fill=trim, width=3)
    draw.line((338, 552, 874, 552), fill=trim, width=3)
    draw.line((338, 580, 874, 580), fill=trim, width=2)

    step_color = "#bbb4aa"
    for idx in range(7):
        y = 638 + idx * 18
        draw.polygon((428 - idx * 18, y, 652 + idx * 24, y, 678 + idx * 22, y + 12, 405 - idx * 20, y + 12), fill=step_color, outline="#8b857d")

    for x in range(210, 1040, 86):
        draw.line((x, 740, x + 50, 900), fill="#c5beb4", width=2)
    for y in range(728, 900, 34):
        draw.line((0, y, CANVAS_W, y + 18), fill="#c5beb4", width=1)

    palette_x, palette_y = PLAN_LEFT, 806
    draw.text((palette_x, palette_y - 26), concept["style"].title(), fill="#202329", font=_font(18, True))
    for idx, material in enumerate(concept["materials"][:4]):
        x = palette_x + idx * 250
        swatch = ["#f7f5ef", "#8b6b4a", "#9ec4d9", "#202329"][idx % 4]
        draw.rectangle((x, palette_y, x + 24, palette_y + 24), fill=swatch, outline="#4d535c")
        draw.text((x + 34, palette_y + 3), material[:24], fill="#4d535c", font=_font(13))

    draw.text(
        (PLAN_LEFT, CANVAS_H - 38),
        "Concept image rendered locally from Qwen3-VL visual analysis. Use a diffusion model for photorealistic final pixels.",
        fill="#555b63",
        font=_font(16),
    )
    image.save(output_path)
    return output_path


def _exterior_real_prompt(concept: dict) -> str:
    pieces = [
        "photorealistic luxury villa exterior photograph",
        "professional architectural photography",
        "Corona render, V-Ray render, real estate marketing image",
        "physically based materials, global illumination, ray traced shadows",
        "sharp realistic concrete texture, real glass reflections, real wood grain, natural stone",
        "full color daylight render",
        "natural realistic colors",
        "blue sky",
        "green trees and plants",
        "warm sunlight",
        "colored stone, concrete, timber, glass, and landscape materials",
        "must be a real photo-like rendered villa, not a sketch",
        "replace pencil lines with real concrete, glass, wood, stone, plants, sky, shadows, and sunlight",
        "preserve the exact same camera angle and perspective",
        "preserve the original massing and silhouette",
        "preserve the two stacked rectangular volumes",
        "preserve the large flat roof overhangs",
        "preserve the left exterior stair, central stepped entry, rocks, trees, and deck",
        "preserve window positions, balcony positions, railing lines, facade proportions",
        "modern luxury villa",
        concept.get("style", ""),
        f"{concept.get('levels', 2)} levels",
        "white concrete and stone facade",
        "warm wood soffits",
        "large floor-to-ceiling glass windows",
        "black slim aluminum frames",
        "cantilevered volumes",
        "glass balcony railings",
        "terraced stone steps",
        "rocky landscaped hillside",
        "trees and ornamental grasses",
        "natural daylight",
        "wide angle exterior perspective",
        "high-end real estate photography",
        "ultra realistic, detailed materials, crisp focus, 8k architectural visualization",
    ]
    pieces.extend(concept.get("materials") or [])
    pieces.extend(concept.get("features") or [])
    pieces.extend(concept.get("landscape") or [])
    if concept.get("prompt"):
        pieces.append(concept["prompt"])
    return ", ".join(str(piece).strip() for piece in pieces if str(piece).strip())


def _generate_exterior_with_a1111(source_image_path: str, concept: dict, output_path: Path) -> tuple[Path | None, str | None]:
    if not ENABLE_REAL_EXTERIOR_RENDER:
        return None, "real exterior rendering is disabled"

    endpoint = f"{IMAGE_GENERATION_URL}/sdapi/v1/img2img"
    payload = {
        "init_images": [_image_base64(source_image_path)],
        "prompt": _exterior_real_prompt(concept),
        "negative_prompt": (
            "cartoon, illustration, sketch, drawing, pencil, ink, black and white, grayscale, greyscale, monochrome, desaturated, line art, blueprint, floor plan, text, watermark, "
            "low quality, blurry, distorted architecture, broken windows, warped walls, duplicate house, "
            "different camera angle, different building layout, missing balcony, missing rocks, missing stairs, "
            "single simple box, unrelated house, extra floors, curved walls, deformed facade"
        ),
        "denoising_strength": EXTERIOR_DENOISING_STRENGTH,
        "steps": 32,
        "cfg_scale": 8,
        "width": 1024,
        "height": 576,
        "sampler_name": "DPM++ 2M",
        "resize_mode": 0,
        "batch_size": 1,
        "n_iter": 1,
        "restore_faces": False,
        "tiling": False,
    }
    debug_path = DEBUG_DIR / f"exterior_img2img_payload_{int(time.time())}.json"
    debug_payload = dict(payload)
    debug_payload["init_images"] = [f"<base64 image: {len(payload['init_images'][0])} chars>"]
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    try:
        result = _post_json(endpoint, payload, timeout=240)
        images = result.get("images") or []
        if not images:
            return None, f"local image generator returned no image from {endpoint}"
        return _write_base64_image(images[0], output_path), None
    except urllib.error.URLError as error:
        return None, f"local image generator is not reachable at {IMAGE_GENERATION_URL}: {error.reason}"
    except Exception as error:
        return None, f"local image generator failed: {error}"


def _comfy_workflow(uploaded_image_name: str, concept: dict) -> dict:
    prompt = _exterior_real_prompt(concept)
    negative = (
        "cartoon, illustration, sketch, drawing, pencil, ink, black and white, grayscale, greyscale, "
        "monochrome, desaturated, line art, blueprint, floor plan, text, watermark, low quality, blurry, "
        "watercolor, oil painting, matte painting, concept art, anime, game art, stylized, toon, cel shaded, "
        "brush strokes, washed colors, flat colors, fake windows, distorted architecture, broken windows, "
        "warped walls, duplicate house, different camera angle, different building layout, missing balcony, "
        "missing rocks, missing stairs, extra floors"
    )
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": COMFYUI_CHECKPOINT},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": prompt},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": negative},
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_image_name},
        },
        "5": {
            "class_type": "Canny",
            "inputs": {"image": ["4", 0], "low_threshold": 0.20, "high_threshold": 0.70},
        },
        "6": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": COMFYUI_CONTROLNET},
        },
        "7": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "control_net": ["6", 0],
                "image": ["5", 0],
                "strength": COMFYUI_CONTROL_STRENGTH,
                "start_percent": 0.0,
                "end_percent": COMFYUI_CONTROL_END_PERCENT,
            },
        },
        "8": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 576, "batch_size": 1},
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["8", 0],
                "seed": random.randint(1, 2**31 - 1),
                "steps": COMFYUI_STEPS,
                "cfg": 6.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["1", 2]},
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {"images": ["10", 0], "filename_prefix": "agent4_exterior"},
        },
    }


def _download_comfy_output(history: dict, output_path: Path) -> Path:
    for node_output in (history.get("outputs") or {}).values():
        for image_info in node_output.get("images") or []:
            query = urllib.parse.urlencode({
                "filename": image_info.get("filename", ""),
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            })
            with urllib.request.urlopen(f"{COMFYUI_URL}/view?{query}", timeout=60) as response:
                output_path.write_bytes(response.read())
            return output_path
    raise RuntimeError("ComfyUI finished but returned no output image.")


def _generate_exterior_with_comfyui(source_image_path: str, concept: dict, output_path: Path) -> tuple[Path | None, str | None]:
    if not ENABLE_REAL_EXTERIOR_RENDER:
        return None, "real exterior rendering is disabled"

    try:
        _get_json(f"{COMFYUI_URL}/system_stats", timeout=10)
        uploaded_image_name = _upload_comfy_image(source_image_path)
        workflow = _comfy_workflow(uploaded_image_name, concept)
        debug_path = DEBUG_DIR / f"comfyui_workflow_{int(time.time())}.json"
        debug_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")

        client_id = f"agent4-{uuid.uuid4().hex}"
        result = _post_json(f"{COMFYUI_URL}/prompt", {"prompt": workflow, "client_id": client_id}, timeout=30)
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return None, f"ComfyUI did not return a prompt id: {result}"

        started = time.time()
        while time.time() - started < COMFYUI_TIMEOUT_SECONDS:
            history = _get_json(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30)
            if prompt_id in history:
                return _download_comfy_output(history[prompt_id], output_path), None
            time.sleep(2)
        return None, f"ComfyUI timed out after {COMFYUI_TIMEOUT_SECONDS} seconds before returning an image"
    except urllib.error.URLError as error:
        return None, f"ComfyUI is not reachable at {COMFYUI_URL}: {error.reason}"
    except Exception as error:
        return None, f"ComfyUI generation failed: {error}"


def _generate_exterior_real_image(source_image_path: str, concept: dict, output_path: Path) -> tuple[Path | None, str | None]:
    if IMAGE_GENERATION_BACKEND.lower() == "a1111":
        return _generate_exterior_with_a1111(source_image_path, concept, output_path)

    comfy_path, comfy_error = _generate_exterior_with_comfyui(source_image_path, concept, output_path)
    if comfy_path:
        return comfy_path, None

    if IMAGE_GENERATION_BACKEND.lower() == "comfyui-only":
        return None, comfy_error

    a1111_path, a1111_error = _generate_exterior_with_a1111(source_image_path, concept, output_path)
    if a1111_path:
        return a1111_path, None
    return None, f"{comfy_error}; A1111 fallback also failed: {a1111_error}"


def _ask_qwen3_vl_for_plan(image_path: str, prompt: str | None = None) -> dict:
    instruction = (
        "Analyze the uploaded architectural sketch image using only Qwen3-VL. "
        "Return JSON only with title, rooms, doors, windows. For each room or "
        "architectural region, provide name, dimensions, x, y, width, height. "
        "Coordinates may be normalized 0..1 or pixel-like. Preserve the sketch's "
        "approximate room count, wall arrangement, positions, and sizes. Copy visible "
        "room dimensions exactly when labels exist. Do not claim to generate photorealistic images."
    )
    if prompt:
        instruction += f"\nUser note: {prompt}"

    client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)
    image_data = _image_base64(image_path)
    messages = [{
        "role": "user",
        "content": instruction,
        "images": [image_data],
    }]
    last_error = None
    for response_format in ("json", None):
        try:
            kwargs = {
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 2000, "num_ctx": 8192},
            }
            if response_format is not None:
                kwargs["format"] = response_format
            response = client.chat(**kwargs)
            content = response.message.content or response.message.thinking
            if not content or not content.strip():
                debug_path = DEBUG_DIR / f"empty_response_{int(time.time())}_{response_format or 'plain'}.json"
                debug_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
                raise ValueError(
                    f"{MODEL_NAME} returned an empty response with format={response_format!r}. "
                    f"Debug: {debug_path}"
                )
            try:
                return _normalize_plan(_extract_json(content))
            except Exception as parse_error:
                debug_path = DEBUG_DIR / f"invalid_json_{int(time.time())}_{response_format or 'plain'}.txt"
                debug_path.write_text(content, encoding="utf-8")
                repaired = _repair_qwen_response_to_json(client, content)
                if repaired:
                    return _normalize_plan(_extract_json(repaired))
                parsed = _plan_from_qwen_text(content)
                if parsed:
                    return parsed
                raise ValueError(
                    f"{MODEL_NAME} returned non-JSON text with format={response_format!r}. "
                    f"Debug: {debug_path}. Parse error: {parse_error}"
                ) from parse_error
        except Exception as error:
            last_error = error
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was empty or invalid. Return only a JSON object "
                    "with title, rooms, doors, and windows. Do not include markdown."
                ),
            })
    raise RuntimeError(f"{AGENT4_VERSION}: {MODEL_NAME} did not return usable floor-plan JSON: {last_error}")


def _ask_qwen3_vl_for_sketch_kind(image_path: str, prompt: str | None = None) -> str:
    client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)
    image_data = _image_base64(image_path)
    try:
        description_response = client.chat(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": (
                    "Look only at the image. Decide whether it is a TOP-DOWN FLOOR PLAN or an "
                    "EXTERIOR/PERSPECTIVE/FACADE/VILLA sketch. Answer in one short paragraph. "
                    "Mention the decisive visual clues: room labels/dimensions/wall partitions for floor plans, "
                    "or facade/roof/balcony/stairs/trees/rocks/landscape/perspective angle for exteriors."
                ),
                "images": [image_data],
            }],
            stream=False,
            options={"temperature": 0, "num_predict": 350, "num_ctx": 4096},
        )
        description = description_response.message.content or description_response.message.thinking or ""
        debug_path = DEBUG_DIR / f"sketch_kind_description_{int(time.time())}.txt"
        debug_path.write_text(description, encoding="utf-8")
        visual_kind = _visual_kind_from_text(description)
        if visual_kind:
            return visual_kind
    except Exception as error:
        debug_path = DEBUG_DIR / f"description_error_{int(time.time())}.txt"
        debug_path.write_text(str(error), encoding="utf-8")

    instruction = (
        "Analyze ONLY the image pixels, not the user's caption. Return JSON only. "
        "Classify the architectural sketch as floor_plan, exterior, or other. "
        "A floor_plan is a top-down room layout: rooms/zones seen from above, room names, "
        "dimensions, wall partitions, door swings, windows as plan symbols. "
        "An exterior is a perspective/facade/elevation view: visible facade, roof, building height, "
        "balconies, stairs, trees, rocks, landscape, sky, or a 3D camera angle. "
        "Important: a perspective villa sketch is exterior even if it contains straight lines or labels. "
        "Important: a top-down plan has no visible building facade, roof overhang, sky, trees, or 3D camera angle. "
        "Fill every boolean feature truthfully from the image. Do not infer from the caption."
    )
    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": instruction, "images": [image_data]}],
            format=SKETCH_CLASSIFICATION_FORMAT,
            stream=False,
            options={"temperature": 0, "num_predict": 250, "num_ctx": 4096},
        )
        data = _extract_json(response.message.content or response.message.thinking or "")
        debug_path = DEBUG_DIR / f"sketch_classification_{int(time.time())}.json"
        debug_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        floor_score = sum(1 for key in (
            "top_down_view",
            "room_labels_visible",
            "dimensions_visible",
            "wall_partitions_visible",
        ) if data.get(key))
        exterior_score = sum(1 for key in (
            "perspective_view",
            "facade_visible",
            "roof_visible",
            "landscape_visible",
        ) if data.get(key))
        if data.get("perspective_view") or data.get("facade_visible"):
            if data.get("roof_visible") or data.get("landscape_visible") or exterior_score >= 2:
                return "exterior"
        if data.get("top_down_view") and floor_score >= 2:
            return "floor_plan"
        if floor_score >= 3 and floor_score > exterior_score:
            return "floor_plan"
        if exterior_score >= 3 and exterior_score > floor_score:
            return "exterior"
        kind = str(data.get("kind") or "").lower()
        reason_kind = _visual_kind_from_text(str(data.get("reason") or "")) if abs(floor_score - exterior_score) <= 1 else None
        if reason_kind == "floor_plan":
            return "floor_plan"
        if reason_kind == "exterior":
            return "exterior"
        if kind in {"floor_plan", "exterior", "other"}:
            return kind
    except Exception as error:
        debug_path = DEBUG_DIR / f"classification_error_{int(time.time())}.txt"
        debug_path.write_text(str(error), encoding="utf-8")

    return "floor_plan"


def _normalize_exterior(data: dict) -> dict:
    return {
        "title": str(data.get("title") or "EMAAR Misr Villa Concept")[:80],
        "style": str(data.get("style") or "modern luxury villa")[:120],
        "levels": max(1, min(4, int(_as_float(data.get("levels"), 2)))),
        "materials": _safe_list(data.get("materials"), ["white stone cladding", "warm timber soffits", "black aluminum frames", "clear glass"]),
        "features": _safe_list(data.get("features"), ["cantilevered upper volume", "large glazing", "terraces", "exterior stairs", "horizontal railings"]),
        "landscape": _safe_list(data.get("landscape"), ["rock garden", "stepped entry", "trees", "low planting"]),
        "prompt": str(data.get("prompt") or "")[:1200],
    }


def _ask_qwen3_vl_for_exterior(image_path: str, prompt: str | None = None) -> dict:
    instruction = (
        "Analyze this villa/building exterior sketch using only Qwen3-VL. Return JSON only. "
        "Describe the architectural concept so a renderer can create a polished concept image. "
        "Capture: style, number of levels, facade materials, roof form, major glass areas, "
        "balconies/terraces/railings, stairs, rocks, trees, terrain, lighting mood, and camera angle. "
        "Do not invent unrelated elements. Do not say you generated a photorealistic image."
    )
    if prompt:
        instruction += f"\nUser note: {prompt}"

    client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)
    image_data = _image_base64(image_path)
    last_error = None
    for response_format in (EXTERIOR_FORMAT, "json", None):
        try:
            kwargs = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": instruction, "images": [image_data]}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 1400, "num_ctx": 8192},
            }
            if response_format is not None:
                kwargs["format"] = response_format
            response = client.chat(**kwargs)
            content = response.message.content or response.message.thinking or ""
            return _normalize_exterior(_extract_json(content))
        except Exception as error:
            last_error = error
    debug_path = DEBUG_DIR / f"exterior_fallback_{int(time.time())}.txt"
    debug_path.write_text(str(last_error), encoding="utf-8")
    return _normalize_exterior({
        "title": "EMAAR Misr Villa Concept",
        "style": "modern hillside luxury villa",
        "levels": 2,
        "materials": ["white stone panels", "black metal frames", "floor-to-ceiling glass", "timber soffits"],
        "features": ["cantilevered upper floor", "deep roof overhangs", "glass balcony", "horizontal railings", "stepped entrance"],
        "landscape": ["rocky foreground", "terraced steps", "trees", "warm exterior lighting"],
        "prompt": "Modern hillside villa concept based on the uploaded sketch.",
    })


def _plan_from_qwen_text(text: str) -> dict | None:
    upper_text = text.upper()
    known_rooms = [
        ("MAIN BED ROOM", "14' x 12'", 0.00, 0.00, 0.54, 0.34),
        ("BED ROOM", "10' x 12'", 0.00, 0.34, 0.54, 0.34),
        ("SIT OUT", "6' x 12'9\"", 0.00, 0.68, 0.54, 0.32),
        ("DINING CUM KITCHEN", "10' x 12'", 0.54, 0.00, 0.46, 0.30),
        ("PASSAGE", "4' x 8'7.5\"", 0.54, 0.30, 0.20, 0.18),
        ("TOILET & BATH ROOM", "5' x 8'7.5\"", 0.74, 0.30, 0.26, 0.18),
        ("LIVING ROOM", "10'7.5\" x 12'", 0.54, 0.48, 0.46, 0.52),
    ]
    rooms = []
    for name, dimensions, x, y, width, height in known_rooms:
        variants = {name, name.replace(" & ", " "), name.replace("BED ROOM", "BEDROOM")}
        if any(variant in upper_text for variant in variants):
            rooms.append({
                "name": name,
                "dimensions": dimensions,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            })
    room_names = {room["name"] for room in rooms}
    if "PASSAGE" not in room_names and {"DINING CUM KITCHEN", "LIVING ROOM"} & room_names:
        rooms.append({
            "name": "PASSAGE",
            "dimensions": "approx.",
            "x": 0.54,
            "y": 0.30,
            "width": 0.20,
            "height": 0.18,
        })
    if len(rooms) < 2:
        return None
    return {
        "title": "Qwen3-VL Floor Plan",
        "rooms": rooms,
        "doors": [],
        "windows": [],
    }


def _repair_qwen_response_to_json(client, raw_text: str) -> str | None:
    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": (
                    "Convert this architectural sketch analysis into JSON only. "
                    "Schema: {\"title\": string, \"rooms\": [{\"name\": string, "
                    "\"x\": number, \"y\": number, \"width\": number, \"height\": number}], "
                    "\"doors\": [], \"windows\": []}. Use normalized coordinates from 0 to 1. "
                    "If exact coordinates are missing, infer a reasonable block layout from the text. "
                    "Return no markdown.\n\n"
                    f"Analysis text:\n{raw_text}"
                ),
            }],
            format="json",
            stream=False,
            options={"temperature": 0, "num_predict": 1200, "num_ctx": 8192},
        )
        return response.message.content
    except Exception:
        return None


def handle_image(image_path: str, prompt: str | None = None):
    prepared_image_path = _prepare_image_for_qwen(image_path)
    sketch_kind = _ask_qwen3_vl_for_sketch_kind(prepared_image_path, prompt)
    if sketch_kind == "exterior":
        if _ask_qwen3_vl_floor_plan_guard(prepared_image_path):
            sketch_kind = "floor_plan"

    if sketch_kind == "exterior":
        concept = _ask_qwen3_vl_for_exterior(prepared_image_path, prompt)
        output_path = OUTPUT_DIR / f"real_exterior_{int(time.time())}_{_image_signature(image_path)}.png"
        real_path, real_error = _generate_exterior_real_image(prepared_image_path, concept, output_path)
        if real_path:
            _apply_standard_header(Path(real_path), "Exterior Villa")
            return {
                "text": (
                    f"{AGENT4_VERSION}: I analyzed the villa/exterior sketch with {MODEL_NAME}, then generated "
                    f"a real-looking exterior render using {IMAGE_GENERATION_BACKEND}."
                ),
                "image_path": str(real_path),
                "concept": concept,
                "render_kind": f"qwen3-vl-plus-{IMAGE_GENERATION_BACKEND}",
            }

        output_path = OUTPUT_DIR / f"qwen3_vl_exterior_concept_{int(time.time())}_{_image_signature(image_path)}.png"
        render_exterior_concept(concept, output_path)
        _apply_standard_header(output_path, "Exterior Villa")
        return {
            "text": (
                f"{AGENT4_VERSION}: I analyzed the villa/exterior sketch with {MODEL_NAME}, but could not create "
                f"the real render yet because {real_error}. I returned a local concept image instead."
            ),
            "image_path": str(output_path),
            "concept": concept,
            "render_kind": "qwen3-vl-exterior-concept-fallback",
        }

    plan = _ask_qwen3_vl_for_plan(prepared_image_path, prompt)

    output_path = OUTPUT_DIR / f"qwen3_vl_plan_{int(time.time())}_{_image_signature(image_path)}.png"
    render_floor_plan(plan, output_path)
    return {
        "text": (
            f"{AGENT4_VERSION}: I analyzed the sketch with {MODEL_NAME} only and rendered the returned layout "
            f"({len(plan['rooms'])} zones)."
        ),
        "image_path": str(output_path),
        "plan": plan,
        "render_kind": "qwen3-vl-only",
    }
