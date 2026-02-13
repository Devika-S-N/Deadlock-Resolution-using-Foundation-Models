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
    yellow_waypoints: List[Tuple[float, float]],  # NEW: all previous waypoints
    retry_note: Optional[str] = None,
) -> str:
    # ========== SOLUTION 1: DEADLOCK DETECTION ==========
    # Count yellow waypoints near the agent (within 2 units)
    local_yellow_count = sum(
        1 for wp in yellow_waypoints 
        if ((wp[0] - agent[0])**2 + (wp[1] - agent[1])**2)**0.5 < 2.0
    )
    
    deadlock_detected = local_yellow_count > 8  # threshold for deadlock
    
    # Calculate escape direction if deadlocked
    escape_instruction = ""
    if deadlock_detected:
        goal_vec = (goal[0] - agent[0], goal[1] - agent[1])
        goal_dist = (goal_vec[0]**2 + goal_vec[1]**2)**0.5
        
        if goal_dist > 0.01:
            # Normalize goal vector
            goal_norm = (goal_vec[0] / goal_dist, goal_vec[1] / goal_dist)
            # Perpendicular directions (90° rotations)
            perp_1 = (-goal_norm[1], goal_norm[0])
            perp_2 = (goal_norm[1], -goal_norm[0])
            
            escape_instruction = f"""
DEADLOCK DETECTED
You have {local_yellow_count} previous waypoints within 2 units - you are STUCK in a loop!

MANDATORY ESCAPE STRATEGY:
1. You MUST break out of this congested area first
2. Move in one of these perpendicular directions to escape: 
   - Direction A: ({perp_1[0]:.2f}, {perp_1[1]:.2f})
   - Direction B: ({perp_2[0]:.2f}, {perp_2[1]:.2f})
3. First 2 waypoints should move AT LEAST 1.5 units away from current position in escape direction
4. Do NOT place any waypoints near existing yellow points
5. After escaping the congested zone, then navigate around the obstacle toward the goal
6. This is CRITICAL - you are currently looping and making no progress!
"""
    
    # ========== SOLUTION 3: SPATIAL AWARENESS ==========
    # Analyze obstacle directions relative to agent
    red_points_above = [p for p in lidar_hits if p[1] > agent[1]]
    red_points_below = [p for p in lidar_hits if p[1] < agent[1]]
    red_points_left = [p for p in lidar_hits if p[0] < agent[0]]
    red_points_right = [p for p in lidar_hits if p[0] > agent[0]]
    
    # Determine goal direction
    goal_dir_x = "RIGHT" if goal[0] > agent[0] else "LEFT"
    goal_dir_y = "UP" if goal[1] > agent[1] else "DOWN"
    
    # Detect if in corridor
    is_corridor = (len(red_points_above) > 3 and len(red_points_below) > 3) or \
                  (len(red_points_left) > 3 and len(red_points_right) > 3)
    
    spatial_context = f"""
SPATIAL SITUATION ANALYSIS:
- Obstacles detected: ABOVE={len(red_points_above)}, BELOW={len(red_points_below)}, LEFT={len(red_points_left)}, RIGHT={len(red_points_right)}
- Environment type: {"CORRIDOR - limited maneuvering space" if is_corridor else "OPEN SPACE - multiple paths possible"}
- Goal direction: {goal_dir_x} and {goal_dir_y}
- Goal is at: {goal}, you are at: {agent}
- Distance to goal: {((goal[0]-agent[0])**2 + (goal[1]-agent[1])**2)**0.5:.2f} units
"""
    
    # Compute forbidden zones (cells with too many visits)
    from collections import defaultdict
    cell_counts = defaultdict(int)
    for wp in yellow_waypoints:
        cell = (int(wp[0]), int(wp[1]))
        cell_counts[cell] += 1
    
    forbidden_cells = [cell for cell, count in cell_counts.items() if count > 3]
    forbidden_zone_str = ""
    if forbidden_cells:
        forbidden_zone_str = f"""
FORBIDDEN ZONES (avoid these grid cells - too many previous visits):
{forbidden_cells[:10]}  # showing first 10
DO NOT place waypoints in or near these cells.
"""
    
    # ========== SHOW RECENT YELLOW WAYPOINTS EXPLICITLY ==========
    # Show the most recent 10 yellow waypoints so VLM can't miss them
    recent_yellow_str = ""
    if yellow_waypoints and len(yellow_waypoints) > 0:
        recent_yellow = yellow_waypoints[-10:] if len(yellow_waypoints) > 10 else yellow_waypoints
        recent_yellow_str = f"""
CRITICAL: RECENT WAYPOINTS TO AVOID (most recent {len(recent_yellow)}):
{recent_yellow}

MANDATORY RULE: DO NOT place waypoints within 1.5 units of these locations!
If you place a waypoint near these coordinates, the agent will OSCILLATE ENDLESSLY!
Example: If recent waypoint is at [5.5, 4.5], do NOT place waypoint at [5.5, 3.0] or [6.0, 4.5].
You MUST choose COMPLETELY DIFFERENT coordinates at least 1.5 units away.
"""
    
    return f"""
You are assisting a mobile robot navigating in a 2D plane to reach a goal while avoiding obstacles and deadlock situations.

IMAGE DESCRIPTION (also depicted visually):
- Green dot: agent at {agent}
- Blue star: goal at {goal}
- Red dots: obstacle points detected by LIDAR
- Yellow dots: previous detour waypoints (areas already explored - AVOID THESE!)

TEXT DATA (authoritative):
- Agent position: {agent}
- Goal position: {goal}
- LIDAR hit points: {lidar_hits}
- Previous waypoint count: {len(yellow_waypoints)}

{escape_instruction}

{spatial_context}

{forbidden_zone_str}

{recent_yellow_str}

TASK:
Return a SMALL SET of LOCAL DETOUR WAYPOINTS to move the agent past the obstacle.
{"PRIORITY: ESCAPE THE DEADLOCK FIRST, then navigate toward goal!" if deadlock_detected else "These waypoints should help the agent make progress toward the goal."}

RULES (MANDATORY):
1. Return ONLY JSON with key "waypoints"
2. Waypoints must be LOCAL (within 3 units of agent, no direct jump to goal)
3. Give exactly 4 waypoints
4. Keep waypoints AT LEAST 1.2 units away from red obstacle points (safety margin for path clearance between waypoints)
5. Keep waypoints AT LEAST 1.0 units away from yellow previous waypoints (critically important!)
6. Do NOT place waypoints in forbidden zones listed above
7. {"ESCAPE DIRECTION FIRST - move perpendicular to goal direction to break out!" if deadlock_detected else "Move tangentially around obstacles, not directly into them"}
8. After clearing the obstacle, gradually turn toward the goal
9. Coordinates must be numeric [x, y] within bounds [0, 15]
10. Think about the spatial situation:
    - If in a NARROW CORRIDOR/TUNNEL: Place waypoints in the CENTER (midpoint between walls), NOT near the walls
    - If obstacles are on both sides (left+right or top+bottom): You're in a corridor - stay centered!
    - If in open space: Go around obstacles with safety margin

CORRIDOR/TUNNEL RULE: If you see obstacles on BOTH sides of the agent, calculate the midpoint between them and place waypoints there!
Example: If obstacles at y=0.3 and y=1.0, place waypoints at y=0.65 (the center), NOT at y=0.5 or y=0.8 (too close to walls).

{f"RETRY FEEDBACK: {retry_note}" if retry_note else ""}

OUTPUT FORMAT (JSON only, no explanation):
{{
  "waypoints": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}}
""".strip()


# ------------------ Core API ------------------
def query_detour_waypoints(
    image_path: str,
    agent: Tuple[float, float],
    goal: Tuple[float, float],
    lidar_hits: List[Tuple[float, float]],
    yellow_waypoints: List[Tuple[float, float]] = None,  # NEW parameter
    model: str = DEFAULT_MODEL,
    retry_note: Optional[str] = None,
    logger = None,  # NEW: Logger for prompt logging
) -> List[Tuple[float, float]]:

    if yellow_waypoints is None:
        yellow_waypoints = []  # default to empty list if not provided

    b64, mime = _encode_image(image_path)
    prompt = _build_prompt(agent, goal, lidar_hits, yellow_waypoints, retry_note)  # pass yellow_waypoints

    # NEW: Log the full prompt if logger provided
    if logger:
        logger.log("=" * 80)
        logger.log("VLM QUERY PROMPT:")
        logger.log("=" * 80)
        logger.log(prompt)
        logger.log("=" * 80)

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