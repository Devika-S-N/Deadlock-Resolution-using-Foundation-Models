import os
import json
from typing import List, Optional

# OpenAI SDK v1.x
# pip install openai==1.*
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

# ===== OpenAI config =====
# Set your API key in the environment:  export OPENAI_API_KEY=...  (Linux/macOS)
# or in Windows PowerShell:  setx OPENAI_API_KEY "<your key>"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # you may try "gpt-4.1-mini" as well

_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    _client = OpenAI(api_key=OPENAI_API_KEY)
else:
    # Defer raising until first use so that module import doesn't crash test environments
    _client = None


def _extract_waypoints(text: str) -> Optional[List[List[float]]]:
    """
    Accepts either:
    ###OUTPUT_START###
    [x, y]
    [x, y]
    ...
    ###OUTPUT_END###
    or a single JSON list: [[x,y], [x,y], ...]
    """
    if not text:
        return None

    # Try to find an explicit JSON list first
    start = text.find("[[")
    if start != -1:
        end = text.find("]]", start)
        if end != -1:
            block = text[start : end + 2]
            try:
                arr = json.loads(block)
                out: List[List[float]] = []
                for pair in arr:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        a, b = pair
                        out.append([float(a), float(b)])
                return out
            except Exception:
                pass

    # Next, try OUTPUT markers with one [x,y] per line
    if "###OUTPUT_START###" in text and "###OUTPUT_END###" in text:
        chunk = text.split("###OUTPUT_START###", 1)[1].split("###OUTPUT_END###", 1)[0]
        lines = [ln.strip().rstrip(",") for ln in chunk.splitlines() if ln.strip()]
        out: List[List[float]] = []
        for ln in lines:
            s = ln.find("["); e = ln.find("]", s + 1)
            if s != -1 and e != -1:
                coords = ln[s + 1 : e].split(",")
                if len(coords) == 2:
                    try:
                        x = float(coords[0].strip()); y = float(coords[1].strip())
                        out.append([x, y])
                    except Exception:
                        continue
        return out if out else None

    return None


def get_detour_waypoints(
    agent_pos,
    goal_pos,
    obstacle_coords,
    k_min: int = 5,
    k_max: int = 8,
    radius: float = 0.6,
    avoid_list: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """
    Drop-in replacement for the Bedrock/Claude version.

    Ask for 5–8 waypoints near the current position that trace the obstacle’s contour.
    Returns a list [[x,y], ...] (may be empty if the LLM fails).
    """
    if _client is None:
        print(
            "Error: OPENAI_API_KEY is not set. Please set it in your environment before calling get_detour_waypoints()."
        )
        return []

    avoid_list = avoid_list or []

    prompt = f"""
You are helping a point robot detour a rectangular or polygonal obstacle in a 2D plane.
The agent is at {agent_pos} and the goal is at {goal_pos}.
The obstacle polygon vertices are: {obstacle_coords} (order as given).
Please output {k_min} to {k_max} detour waypoints that trace along the obstacle's *outside* contour,
each within ~{radius} units of the previous point (start from the agent position’s nearest point on the obstacle contour),
and keep them inside the world bounds [0,10]x[0,10]. Do not enter the obstacle.

Hard rules:
- Keep ~{radius} step between consecutive waypoints (±0.2 ok).
- Never place a waypoint inside the obstacle.
- Move consistently along the same side of the obstacle (no ping-pong).
- Prefer turning around corners rather than cutting through.
- Avoid any of these previously tried points: {avoid_list}

Return ONLY the waypoints in one of these formats:

Option A:
###OUTPUT_START###
[x1, y1]
[x2, y2]
...
###OUTPUT_END###

Option B (single JSON line):
[[x1,y1],[x2,y2],...]
"""

    try:
        # Chat Completions API (stable in openai 1.x)
        resp: ChatCompletion = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        text = resp.choices[0].message.content if resp.choices else ""

        wps = _extract_waypoints(text) or []

        # Deduplicate & sanitize
        cleaned: List[List[float]] = []
        seen = set()
        for wp in wps:
            try:
                x, y = float(wp[0]), float(wp[1])
            except Exception:
                continue
            key = (round(x, 4), round(y, 4))
            if key not in seen:
                seen.add(key)
                cleaned.append([x, y])
        return cleaned

    except Exception as e:
        print(f"Error querying OpenAI model: {e}")
        return []
