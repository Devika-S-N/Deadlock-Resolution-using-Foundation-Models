import boto3
import json
import os
from botocore.exceptions import BotoCoreError, ClientError

# ===== Bedrock config =====
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
AWS_PROFILE = "MyProfile1"
AWS_REGION = "us-east-1"

os.environ["AWS_PROFILE"] = AWS_PROFILE
os.environ["AWS_DEFAULT_REGION"] = AWS_REGION
client = boto3.client("bedrock-runtime")

def _extract_waypoints(text: str):
    """
    Accepts either:
    ###OUTPUT_START###
    [x, y]
    [x, y]
    ...
    ###OUTPUT_END###
    or a single JSON list: [[x,y], [x,y], ...]
    """
    # First try to find an explicit JSON list
    start = text.find("[[")
    if start != -1:
        end = text.find("]]", start)
        if end != -1:
            block = text[start:end+2]
            try:
                arr = json.loads(block)
                return [[float(a), float(b)] for a, b in arr if len([a,b]) == 2]
            except Exception:
                pass

    # Next, try OUTPUT markers with one [x,y] per line
    if "###OUTPUT_START###" in text and "###OUTPUT_END###" in text:
        chunk = text.split("###OUTPUT_START###", 1)[1].split("###OUTPUT_END###", 1)[0]
        lines = [ln.strip().rstrip(",") for ln in chunk.splitlines() if ln.strip()]
        out = []
        for ln in lines:
            s = ln.find("["); e = ln.find("]", s+1)
            if s != -1 and e != -1:
                coords = ln[s+1:e].split(",")
                if len(coords) == 2:
                    try:
                        x = float(coords[0].strip()); y = float(coords[1].strip())
                        out.append([x, y])
                    except Exception:
                        continue
        return out if out else None

    return None

def get_detour_waypoints(agent_pos, goal_pos, obstacle_coords, k_min=5, k_max=8, radius=0.6, avoid_list=None):
    """
    Ask for 5–8 waypoints near current position that trace the obstacle’s contour.
    Returns a list [[x,y], ...] (may be empty if LLM fails).
    """
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
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 400,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        }
        resp = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        payload = json.loads(resp["body"].read())
        text = payload["content"][0]["text"]
        wps = _extract_waypoints(text) or []
        # Dedup & sanitize
        cleaned = []
        seen = set()
        for wp in wps:
            key = (round(float(wp[0]), 4), round(float(wp[1]), 4))
            if key not in seen:
                seen.add(key)
                cleaned.append([float(wp[0]), float(wp[1])])
        return cleaned
    except (BotoCoreError, ClientError, KeyError, json.JSONDecodeError) as e:
        print(f"Error querying LLM: {e}")
        return []