# extract_scene_aws.py
# Library: call Anthropic Claude on AWS Bedrock to extract scene JSON.
# Public API:
#   extract_scene_from_image(image_path: str, model: str = "haiku") -> dict
#
# Models: "haiku" | "sonnet"
# Credentials: uses AWS profile + region exactly like your earlier pattern.

import os
import json
import base64
from pathlib import Path
from typing import List, Tuple

import boto3
from botocore.exceptions import (
    BotoCoreError, ClientError, NoCredentialsError, ProfileNotFound
)

# ------------------ Model IDs ------------------
MODEL_MAP = {
    "haiku":  "anthropic.claude-3-haiku-20240307-v1:0",
    "sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
}

# ===== Bedrock config (your pattern) =====
MODEL_DEFAULT = "haiku"
AWS_PROFILE   = "MyProfile1"
AWS_REGION    = "us-east-1"

os.environ["AWS_PROFILE"]        = AWS_PROFILE
os.environ["AWS_DEFAULT_REGION"] = AWS_REGION

def _make_bedrock_client():
    try:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        return session.client("bedrock-runtime", region_name=AWS_REGION)
    except ProfileNotFound as e:
        raise RuntimeError(
            f"AWS profile '{AWS_PROFILE}' not found. Configure it with `aws configure --profile {AWS_PROFILE}`."
        ) from e
    except NoCredentialsError as e:
        raise RuntimeError("Unable to locate credentials for the specified AWS profile.") from e

_bedrock = None
def _client():
    global _bedrock
    if _bedrock is None:
        _bedrock = _make_bedrock_client()
    return _bedrock

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
SELF-AUDIT (SILENT)
Before producing the JSON:
• Verify every coordinate ∈ S and within [0,10].
• For rectangles, confirm exactly 2 unique x’s and 2 unique y’s. If 3 appear, adjust by snapping INWARD to the nearest value in S that lies on the observed black boundary.
• Ensure consecutive edges alternate horizontal/vertical.
• Reject any boundary that would include white pixels; shrink inward one step if necessary.
• Confirm the first vertex is bottom-left and ordering is CCW.

FINAL OUTPUT
Return ONLY this compact JSON between markers:

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

GOAL
Return ONLY the final JSON (at the end) with agent, goal, and obstacle vertices.
Vertices must start at the BOTTOM-LEFT corner and proceed COUNTER-CLOCKWISE (CCW).
Do not include any commentary with the JSON; place it strictly between ###OUTPUT_START### and ###OUTPUT_END###.

DISCRETE COORDINATE SET
All coordinates MUST be chosen from the set S = [0.0, 0.5, 1.0, 1.5, . . . . , 9.5, 10.0. Never output values outside [0,10].

BOUNDARY READING RULES (CONSERVATIVE)
1) Determine each obstacle boundary EXACTLY from the visible black pixels.
2) When a black edge visually lies between two gridlines, SNAP INWARD to the nearest value in S that is still fully inside the black region (never outward).
   Examples:
   • If the top edge touches y=3.0, the reported top must be 3.0 (NOT 3.5 or 4.0).
   • If any edge appears ambiguous by ≤0.5, choose the inward value (0.5 toward the interior), never expand.
3) Do NOT merge separate black regions if a white gap ≥ 0.4 units exists.

VERTEX CONSTRUCTION
A) Rectangles: use exactly 4 vertices with 2 distinct x-values and 2 distinct y-values.
   Order: (xmin,ymin) → (xmax,ymin) → (xmax,ymax) → (xmin,ymax).
B) Other rectilinear polygons: return the minimal vertex chain without repeating the first vertex at the end.
C) After listing vertices, ensure CCW orientation; if clockwise, reverse order (keeping the first vertex fixed).

AGENT & GOAL
Read directly from markers (blue dot, green star). Choose the nearest value in S for x and y.

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
from typing import List
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

# ------------------ Bedrock call ------------------
def _call_bedrock(image_path: str, model_key: str) -> dict:
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model '{model_key}'. Choose from {list(MODEL_MAP)}")
    model_id = MODEL_MAP[model_key]

    b64, mime = _encode_image_b64(image_path)
    prompt = _make_prompt()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}
            ]}
        ]
    }

    try:
        resp = _client().invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        text = payload["content"][0]["text"]
    except (BotoCoreError, ClientError, KeyError, json.JSONDecodeError, IndexError) as e:
        raise RuntimeError(f"Bedrock error: {e}")

    raw = _extract_json_from_text(text)
    return _validate_and_normalize_scene(raw)

# ------------------ Public API ------------------
def extract_scene_from_image(image_path: str, model: str = MODEL_DEFAULT) -> dict:
    """Return normalized scene dict for the given image via Claude (AWS Bedrock)."""
    return _call_bedrock(image_path, model_key=model)
