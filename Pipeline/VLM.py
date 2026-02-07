# VLM.py
# Vision-Language Model interface for local detour waypoint generation

import os
import base64
import json
from pathlib import Path
from typing import List, Tuple, Optional
from openai import OpenAI

# ------------------ Setup ------------------
DEFAULT_MODEL = "gpt-5.2"   # or "gpt-4o", "gpt-4o-mini", "gpt-5"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set.")

client = OpenAI(api_key=api_key)

# ------------------ Helpers ------------------
def _encode_image(image_path: str) -> Tuple[str, str]:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(image_path)

    ext = p.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return base64.b64encode(p.read_bytes()).decode("utf-8"), mime


def _build_prompt(
    agent: Tuple[float, float],
    goal: Tuple[float, float],
    lidar_hits: List[Tuple[float, float]],
    retry_note: Optional[str] = None,
) -> str:
    return f"""
You are assisting a mobile robot navigating in a 2D plane to reach a goal while avoiding obstacles and deadlock situations.

IMAGE DESCRIPTION (also depicted visually):
- Green dot: agent at {agent}
- Blue star: goal at {goal}
- Red dots: obstacle points detected by LIDAR
- Black dots: previous detour waypoints taken by the agent

TEXT DATA (authoritative):
- Agent position: {agent}
- Goal position: {goal}
- LIDAR hit points: {lidar_hits}

TASK:
Return a SMALL SET of LOCAL DETOUR WAYPOINTS to move the agent past the obstacle.
These waypoints should help the agent make progress toward the goal.
If you see too many black points close together, treat that area as a deadlock zone.
Try to navigate away from dense black clusters and red points, while moving toward the goal.

RULES (MANDATORY):
1. Return ONLY JSON.
2. Output key: "waypoints"
3. Waypoints must be LOCAL (close to the agent, no direct jump to goal).
4. 3–6 waypoints maximum.
5. Do NOT place waypoints on red points.
6. Do NOT assume any unseen obstacles.
7. The new waypoints should be atleast 0.5 units aways from the previous waypoints marked in black.
8. Always try to keep the waypoints atelast 0.5 units away from the red dots (obstacle points).
9. The waypoints should help the agent move closer to the goal.
10. Always try to move away from the red dots (obstacle points).
11. If you are trapped in a corner, navigate aaway from the corner first before moving towards the goal.
12. Try to explore new directions if you are unable to find a path towards the goal.
13. Avoid repeating or overlapping with already visitied black points.
14. Once you encounter an obstacle, you priority must be to get past that obstacle before moving towards the goal.
15. Make sure the waypoints doesnt make the agent get stuck in the same area again.
14. Coordinates must be numeric [x, y].

{f"NOTE: {retry_note}" if retry_note else ""}

OUTPUT FORMAT:
{{
  "waypoints": [[x1, y1], [x2, y2], ...]
}}
""".strip()


# ------------------ Core API ------------------
def query_detour_waypoints(
    image_path: str,
    agent: Tuple[float, float],
    goal: Tuple[float, float],
    lidar_hits: List[Tuple[float, float]],
    model: str = DEFAULT_MODEL,
    retry_note: Optional[str] = None,
) -> List[Tuple[float, float]]:

    b64, mime = _encode_image(image_path)
    prompt = _build_prompt(agent, goal, lidar_hits, retry_note)

    if model in ("gpt-4o", "gpt-4o-mini"):
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}}
                ]
            }],
            temperature=0.4,
            max_tokens=300,
        )
        text = resp.choices[0].message.content
    else:  # gpt-5 using new Responses API
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Please analyze the image and return only the JSON response."},
                        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}
                    ]
                }
            ],
            reasoning={"effort": "minimal"},  #  minimal reasoning mode
        )

        text = resp.output_text



    # ---- parse JSON ----
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        wps = data.get("waypoints", [])
        return [tuple(map(float, wp)) for wp in wps]
    except Exception as e:
        raise RuntimeError(f"Failed to parse VLM output: {e}")
