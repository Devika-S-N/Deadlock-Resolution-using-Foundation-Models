from utils.llm_interface import LLMInterface
from utils.geometry import get_relevant_obstacles, waypoint_clear_of_obstacle, line_intersects_obstacle
import logging
import math

class LocalPlanner:
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.logger = logging.getLogger('path_planner')
        self.safety_buffer = 0.5  # Minimum distance from obstacles
        
    def describe_obstacle(self, obstacle):
        """Convert obstacle to corner points with safety buffer"""
        ox, oy, w, h = obstacle
        r = self.safety_buffer
        return [
            (ox - r, oy - r),  # Bottom-left
            (ox + w + r, oy - r),  # Bottom-right
            (ox + w + r, oy + h + r),  # Top-right
            (ox - r, oy + h + r)   # Top-left
        ]

    def generate_detour_prompt(self, p1, p2, obstacles):
        """Generate prompt with explicit routing instructions around obstacles"""
        obstacle_descriptions = []
        for i, obs in enumerate(obstacles, 1):
            ox, oy, w, h = obs
            r = self.safety_buffer
            obstacle_descriptions.append(
                f"Obstacle {i} (avoid by {r} units):\n"
                f"- Original: {ox, oy, w, h}\n"
                f"- Safe path must go around these edges:\n"
                f"  Left: x = {ox-r} to {ox-r}, y = {oy-r} to {oy+h+r}\n"
                f"  Right: x = {ox+w+r} to {ox+w+r}, y = {oy-r} to {oy+h+r}\n"
                f"  Bottom: y = {oy-r} to {oy-r}, x = {ox-r} to {ox+w+r}\n"
                f"  Top: y = {oy+h+r} to {oy+h+r}, x = {ox-r} to {ox+w+r}"
            )
        
        prompt = (
            f"Generate a detour path from {p1} to {p2} that routes AROUND obstacles.\n"
            f"Critical Requirements:\n"
            f"1. Stay exactly {self.safety_buffer} units away from obstacle edges\n"
            f"2. Generate 5-7 waypoints that follow obstacle contours\n"
            f"3. Each segment ≤0.2 units\n"
            f"4. Choose shortest safe path around obstacles\n"
            f"5. The detour waypoints should be generted strictly between the two given start and end point. DO NOT generate random intermediate points\n\n"
            f"Obstacle Details (follow edges at {self.safety_buffer} units distance):\n"
            + "="*50 + "\n"
            + "\n".join(obstacle_descriptions) + "\n"
            + "="*50 + "\n\n"
            f"Output ONLY this format with waypoints tracing obstacle edges:\n"
            f"###OUTPUT_START###\n[[x1,y1],[x2,y2],...]\n###OUTPUT_END###\n"
            f"Example for left edge routing: [[{ox-r},{oy-r}, [{ox-r},{oy+h/2}], [{ox-r},{oy+h+r}]]"
        )
        return prompt

    def validate_detour(self, waypoints, obstacles, p1, p2):
        """Strict validation of edge-following waypoints"""
        full_path = [p1] + waypoints + [p2]
        
        # Check path doesn't intersect obstacles
        for i in range(len(full_path)-1):
            for obs in obstacles:
                if line_intersects_obstacle(full_path[i], full_path[i+1], obs):
                    return False, f"Path still intersects obstacle {obs}"
        
        return True, "Validation passed"

    def plan_local_path(self, p1, p2, obstacles, max_retries=20):
        """Generate local detour with detailed obstacle descriptions"""
        self.logger.info("\n%s", '='*100)
        self.logger.info("Generating detour between %s->%s with %d obstacles", p1, p2, len(obstacles))
        
        for attempt in range(1, max_retries + 1):
            self.logger.info("\n\n ----> Attempt %d/%d", attempt, max_retries)
            print(f"-->local path planning {attempt}/{max_retries}")
            
            try:
                prompt = self.generate_detour_prompt(p1, p2, obstacles)
                self.logger.debug("Prompt:\n%s", prompt)
                
                response = self.llm.query_llm(prompt, temperature=0.1, max_tokens=350)
                self.logger.debug("Raw Response:\n%s", response)
                
                waypoints = self.llm.extract_waypoints(response)
                self.logger.debug("Extracted Waypoints: %s", waypoints)
                
                is_valid, validation_msg = self.validate_detour(waypoints, obstacles, p1, p2)
                
                if not is_valid:
                    self.logger.warning("Validation failed: %s", validation_msg)
                    continue
                    
                self.logger.info(
                    "Successful detour on attempt %d\nGenerated %d valid waypoints:\n%s",
                    attempt, len(waypoints), waypoints
                )
                return waypoints
                
            except Exception as e:
                self.logger.error("Attempt %d failed: %s", attempt, str(e))
                if attempt == max_retries:
                    self.logger.critical("All attempts failed for %s->%s", p1, p2)
                    raise RuntimeError(
                        f"Failed to generate valid detour after {max_retries} attempts\n"
                        f"Last error: {str(e)}"
                    )
                continue

    def refine_global_path(self, waypoints, obstacles):
        """Process global path and insert detours where needed"""
        if not waypoints:
            return []
            
        refined_path = [waypoints[0]]
        
        for i in range(len(waypoints)-1):
            p1, p2 = waypoints[i], waypoints[i+1]
            
            # Get obstacles that intersect this segment
            relevant_obs = [obs for obs in obstacles if line_intersects_obstacle(p1, p2, obs)]
            
            if relevant_obs:
                self.logger.info("\nObstacle detected between %s->%s", p1, p2)
                self.logger.info("Relevant obstacles: %s", relevant_obs)
                
                try:
                    detour = self.plan_local_path(p1, p2, relevant_obs)
                    refined_path.extend(detour)
                    self.logger.info("Added %d detour points", len(detour))
                except Exception as e:
                    self.logger.error("Using fallback path due to: %s", str(e))
                    refined_path.append(p2)
            else:
                refined_path.append(p2)
                
        return refined_path