## local_planner.py (final fix: wrap multiline output as list)
import boto3
import json
import re
import time
import os
import numpy as np
from shapely.geometry import LineString, box, Point
from typing import List, Tuple
import logging

class LocalPlanner:
    def __init__(self, model_id="anthropic.claude-3-haiku-20240307-v1:0",
                 aws_profile="MyProfile1", aws_region="us-east-1"):
        try:
            os.environ['AWS_PROFILE'] = aws_profile
            os.environ['AWS_DEFAULT_REGION'] = aws_region
            self.bedrock = boto3.client("bedrock-runtime")
            self.model_id = model_id
            self.logger = logging.getLogger("LocalPlanner")
            self.logger.info("AWS Bedrock Runtime client initialized")
        except Exception as e:
            self.logger.error(f"AWS initialization failed: {str(e)}")
            raise

    def extract_waypoints(self, response_text: str) -> List[Tuple[float, float]]:
        try:
            print("=== LLM Raw Output ===")
            print(response_text)
            print("======================")

            match = re.search(r"###OUTPUT_START###(.*?)###OUTPUT_END###", response_text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                content = content.replace("(", "[").replace(")", "]")

                # Normalize each line into a list
                lines = [line.strip().rstrip(',') for line in content.splitlines() if line.strip()]
                normalized_lines = []
                for line in lines:
                    if not line.startswith("["):
                        line = "[" + line + "]"
                    normalized_lines.append(line)

                joined = "[" + ", ".join(normalized_lines) + "]"
                return json.loads(joined)

            list_match = re.search(r"\[[^\[\]]+\]", response_text)
            if list_match:
                return eval(list_match.group(0))
        except Exception as e:
            self.logger.error(f"Waypoint extraction failed: {str(e)}")
        return None





    def generate_detour(self, current_pos: Tuple[float, float],
                        collision_point: Tuple[float, float],
                        next_waypoint: Tuple[float, float],
                        goal: Tuple[float, float],
                        obstacle: Tuple[float, float, float, float],
                        buffer: float = 0.5) -> List[Tuple[float, float]]:
        ox, oy, w, h = obstacle
        edge = self._identify_collision_edge(collision_point, obstacle)

        boundary = [
            (ox-buffer, oy-buffer),
            (ox-buffer, oy+h+buffer),
            (ox+w+buffer, oy+h+buffer),
            (ox+w+buffer, oy-buffer)
        ]

        prompt = f"""
        Environment bounds: x and y in [0,10]
        Generate 3-5 detour waypoints from {current_pos} to {next_waypoint}:
        - Start near but safely away from the collision point: {collision_point}
        - Follow the {edge} edge of obstacle at {obstacle}
        - Maintain {buffer} units clearance
        - Avoid obstacle and stay within bounds
        - Generate the waypoints such that line joining the adjacent waypoint should be paraller to the obstacle edge.
        - Output exactly 10 points between ###OUTPUT_START### and ###OUTPUT_END###

        Boundary points (buffered): {boundary}
        Movement direction: {self._get_direction(current_pos, next_waypoint)}
        """

        for attempt in range(3):
            try:
                raw_waypoints = self.query_llm(prompt)
                if not raw_waypoints:
                    continue

                if self._validate_detour(current_pos, next_waypoint, raw_waypoints, obstacle, buffer):
                    return raw_waypoints

            except Exception as e:
                self.logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(1)

        raise RuntimeError("LLM failed to generate valid detour waypoints after 3 attempts.")

    def query_llm(self, prompt: str) -> List[Tuple[float, float]]:
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            })
        )
        response_text = json.loads(response["body"].read())["content"][0]["text"]
        return self.extract_waypoints(response_text)


    def _identify_collision_edge(self, point: Tuple[float, float], 
                               obstacle: Tuple[float, float, float, float]) -> str:
        """Identifies which edge of obstacle was hit"""
        ox, oy, w, h = obstacle
        x, y = point
        
        if abs(x - ox) < 0.1: return "left"
        if abs(x - (ox + w)) < 0.1: return "right"
        if abs(y - oy) < 0.1: return "bottom"
        if abs(y - (oy + h)) < 0.1: return "top"
        return "unknown"

    def _generate_edge_detour(self, current_pos: Tuple[float, float],
                            collision_point: Tuple[float, float],
                            next_waypoint: Tuple[float, float],
                            obstacle: Tuple[float, float, float, float],
                            buffer: float,
                            edge: str) -> List[Tuple[float, float]]:
        ox, oy, w, h = obstacle
        detour = []

        if edge == "left":
            detour.append((ox - buffer, collision_point[1]))
            detour.append((ox - buffer, oy + h + buffer))
            detour.append((ox + w / 2, oy + h + buffer))
        elif edge == "right":
            detour.append((ox + w + buffer, collision_point[1]))
            detour.append((ox + w + buffer, oy - buffer))
            detour.append((ox + w / 2, oy - buffer))
        elif edge == "top":
            detour.append((collision_point[0], oy + h + buffer))
            detour.append((ox + w + buffer, oy + h + buffer))
            detour.append((ox + w + buffer, oy + h / 2))
        elif edge == "bottom":
            detour.append((collision_point[0], oy - buffer))
            detour.append((ox - buffer, oy - buffer))
            detour.append((ox - buffer, oy + h / 2))
        else:
            # Default to a direct buffer offset if edge is unknown
            detour.append((current_pos[0], current_pos[1] + buffer))
            detour.append((current_pos[0] + buffer, current_pos[1] + buffer))

        detour.append(next_waypoint)
        return detour

    def _validate_detour(self, start: Tuple[float, float],
                         end: Tuple[float, float],
                         waypoints: List[Tuple[float, float]],
                         obstacle: Tuple[float, float, float, float],
                         buffer: float) -> bool:
        if not waypoints:
            return False

        path = waypoints + [end]
        ox, oy, w, h = obstacle
        obstacle_box = box(ox - buffer, oy - buffer, ox + w + buffer, oy + h + buffer)

        for i in range(len(path) - 1):
            segment = LineString([path[i], path[i + 1]])
            if segment.intersects(obstacle_box):
                print(f"Segment {path[i]} → {path[i+1]} intersects obstacle box {obstacle_box.bounds}")
                return False

        return True

    def _get_direction(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> str:
        """Returns movement direction between points"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "up" if dy > 0 else "down"

    def _is_waypoint_safe(self, waypoint: Tuple[float, float], 
                        obstacles: List[Tuple[float, float, float, float]], 
                        buffer: float) -> bool:
        """Check if waypoint maintains safe distance from obstacles"""
        for ox, oy, w, h in obstacles:
            if (ox - buffer <= waypoint[0] <= ox + w + buffer and 
                oy - buffer <= waypoint[1] <= oy + h + buffer):
                return False
        return True


    def _segment_intersects_obstacle(self, p1: Tuple[float, float], 
                                   p2: Tuple[float, float], 
                                   obstacle: Tuple[float, float, float, float]) -> bool:
        """Check if line segment intersects an obstacle"""
        line = LineString([p1, p2])
        ox, oy, w, h = obstacle
        return line.intersects(box(ox, oy, ox + w, oy + h))