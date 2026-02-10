# extract_scene_gpt.py
# Library: call OpenAI GPT (gpt-4o, gpt-4o-mini, gpt-5) to extract scene JSON.
# Public API:
#   extract_scene_from_image(image_path: str, model: str = "gpt-5") -> dict
#
# Credentials: export OPENAI_API_KEY=...

import os
import json
import base64
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI

OPENAI_MODEL_DEFAULT = "gpt-5"  # or "gpt-4o", "gpt-4o-mini"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set.")
client = OpenAI(api_key=api_key)

# ------------------ IO helpers ------------------
def _encode_image_b64(image_path: str) -> Tuple[str, str]:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    ext = p.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        raise ValueError("Image must be .png or .jpg/.jpeg")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return base64.b64encode(p.read_bytes()).decode("utf-8"), mime

# ------------------ Prompt/schema ------------------
def _strict_schema() -> str:
    return """
###OUTPUT_START###
{
  "agent": [x, y],
  "goal": [x, y],
  "obstacles": [
    {"id": "obs_1", "vertices": [[x1,y1],[x2,y2],...]},
    {"id": "obs_2", "vertices": [[x1,y1],[x2,y2],...]}
  ]
}
###OUTPUT_END###
""".strip()

def _make_prompt() -> str:
    return f"""
You will see an image of a 2D 10×10 world on a white background.
- Obstacles are SOLID BLACK, axis-aligned rectangles or rectilinear polygons (horizontal/vertical edges only).
- The agent is a BLUE dot; the goal is a GREEN star. Major gridlines are at integer coordinates.
- Origin (0,0) is bottom-left; y increases upward.

Return ONLY the final JSON with agent, goal, and obstacle vertices.
- Vertices must start at the BOTTOM-LEFT corner and proceed COUNTER-CLOCKWISE (CCW).
- All coordinates must be within [0,10] and chosen from {0.0,0.5,1.0,...,9.5,10.0}.
- Never extend vertices beyond the visible black region.
- Do not include commentary, only the JSON between markers.

{_strict_schema()}
""".strip()

def _extract_json_from_text(text: str) -> dict:
    if "###OUTPUT_START###" in text and "###OUTPUT_END###" in text:
        text = text.split("###OUTPUT_START###", 1)[1].split("###OUTPUT_END###", 1)[0].strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    raise ValueError("Unclosed JSON object in model response.")

# ------------------ Geometry + normalization ------------------
def _snap05(x: float) -> float:
    return round(float(x) * 2.0) / 2.0

def _signed_area(vertices: List[List[float]]) -> float:
    area = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area

def _rotate_to_start(vertices: List[List[float]], start_idx: int) -> List[List[float]]:
    return vertices[start_idx:] + vertices[:start_idx]

def _bottom_left_index(vertices: List[List[float]]) -> int:
    return min(range(len(vertices)), key=lambda i: (vertices[i][1], vertices[i][0]))

def _ensure_ccw_and_bottom_left(vertices: List[List[float]]) -> List[List[float]]:
    if len(vertices) < 3:
        return vertices[:]
    if vertices[0] == vertices[-1]:
        vertices = vertices[:-1]
    bl = _bottom_left_index(vertices)
    v = _rotate_to_start(vertices, bl)
    if _signed_area(v) < 0:
        v = [v[0]] + list(reversed(v[1:]))
    return v

def _validate_and_normalize_scene(scene: dict) -> dict:
    if not isinstance(scene, dict):
        raise ValueError("Scene must be a JSON object.")
    for k in ("agent", "goal", "obstacles"):
        if k not in scene:
            raise ValueError(f"Missing key: {k}")

    def _xy(v):
        return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)

    if not _xy(scene["agent"]) or not _xy(scene["goal"]):
        raise ValueError("agent/goal must be [x,y] numbers.")

    ax, ay = _snap05(scene["agent"][0]), _snap05(scene["agent"][1])
    gx, gy = _snap05(scene["goal"][0]),  _snap05(scene["goal"][1])
    for name, (x, y) in (("agent", (ax, ay)), ("goal", (gx, gy))):
        if not (0.0 <= x <= 10.0 and 0.0 <= y <= 10.0):
            raise ValueError(f"{name} out of bounds: {[x, y]}")

    obstacles = scene["obstacles"]
    if not isinstance(obstacles, list):
        raise ValueError("obstacles must be a list.")

    norm_obs = []
    for i, ob in enumerate(obstacles, start=1):
        if not isinstance(ob, dict):
            raise ValueError(f"obstacle #{i} must be an object.")
        oid = ob.get("id", f"obs_{i}")
        verts = ob.get("vertices")
        if not (isinstance(verts, list) and len(verts) >= 3):
            raise ValueError(f"{oid} must have >=3 vertices.")

        snapped = []
        for v in verts:
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise ValueError(f"{oid} has a non-[x,y] vertex: {v}")
            x, y = _snap05(v[0]), _snap05(v[1])
            if not (0.0 <= x <= 10.0 and 0.0 <= y <= 10.0):
                raise ValueError(f"{oid} vertex out of bounds: {[x, y]}")
            snapped.append([x, y])

        snapped = _ensure_ccw_and_bottom_left(snapped)
        norm_obs.append({"id": str(oid), "vertices": snapped})

    return {"agent": [ax, ay], "goal": [gx, gy], "obstacles": norm_obs}

# ------------------ OpenAI call ------------------
def _call_openai(image_path: str, model: str) -> dict:
    b64, mime = _encode_image_b64(image_path)
    prompt = _make_prompt()

    try:
        if model in ("gpt-4o", "gpt-4o-mini"):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    ]}
                ],
                max_tokens=800,
                temperature=0.0,
            )
        else:  # gpt-5 (no max_tokens / temperature)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    ]}
                ]
            )
        text = resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")

    raw = _extract_json_from_text(text)
    return _validate_and_normalize_scene(raw)


# ------------------ Public API ------------------
def extract_scene_from_image(image_path: str, model: str = OPENAI_MODEL_DEFAULT) -> dict:
    """Return normalized scene dict for the given image via OpenAI (gpt-4o, gpt-4o-mini, gpt-5)."""
    if model not in ("gpt-4o", "gpt-4o-mini", "gpt-5"):
        raise ValueError("model must be 'gpt-4o', 'gpt-4o-mini', or 'gpt-5'")
    return _call_openai(image_path, model=model)
