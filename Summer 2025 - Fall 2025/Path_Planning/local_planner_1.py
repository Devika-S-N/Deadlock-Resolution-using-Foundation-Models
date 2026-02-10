from utils.llm_interface import LLMInterface
from utils.geometry import get_relevant_obstacles, waypoint_clear_of_obstacle
import logging

class LocalPlanner:
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.logger = logging.getLogger('path_planner')
        
    def generate_detour_prompt(self, p1, p2, relevant_obstacles, r=0.1):
        """Generate prompt for obstacle avoidance"""
        prompt = (
            f"You are generating a detour path from Start: {p1} to Goal: {p2}.\n"
            f"Nearby obstacles (x, y, width, height): {relevant_obstacles}.\n"
            f"Maintain a safety buffer of at least {r} units from all obstacles.\n"
            f"Generate a safe, curved path around the obstacles.\n"
            f"Output at least 5 smooth waypoints in this format:\n"
            f"###OUTPUT_START###\n[[x1, y1], [x2, y2], ...]\n###OUTPUT_END###"
        )
        self.logger.debug(f"Generated detour prompt for segment {p1}->{p2}")
        return prompt

    def plan_local_path(self, p1, p2, obstacles, max_retries=20):
        """Generate local detour around obstacles"""
        self.logger.info(f"Starting local planning for segment {p1}->{p2}")
        relevant_obstacles = get_relevant_obstacles(p1, p2, obstacles)
        self.logger.debug(f"Relevant obstacles: {relevant_obstacles}")
        
        for attempt in range(1, max_retries + 1):
            self.logger.info(f"Attempt {attempt}/{max_retries} for segment {p1}->{p2}")
            
            try:
                # Generate and log prompt
                prompt = self.generate_detour_prompt(p1, p2, relevant_obstacles)
                self.logger.debug(f"Attempt {attempt} Prompt:\n{prompt}")
                
                # Query LLM and log response
                response = self.llm.query_llm(prompt, temperature=0.2, max_tokens=250)
                self.logger.debug(f"Attempt {attempt} Raw Response:\n{response}")
                
                # Extract and validate waypoints
                waypoints = self.llm.extract_waypoints(response)
                self.logger.debug(f"Attempt {attempt} Extracted Waypoints: {waypoints}")
                
                if not waypoints:
                    self.logger.warning(f"Attempt {attempt} returned no waypoints")
                    continue
                    
                if len(waypoints) < 3:
                    self.logger.warning(f"Attempt {attempt} returned only {len(waypoints)} waypoints (minimum 3 required)")
                    continue
                
                # Validate waypoint clearance
                invalid_waypoints = []
                valid = True
                for x, y in waypoints:
                    for ob in relevant_obstacles:
                        if not waypoint_clear_of_obstacle(x, y, ob):
                            invalid_waypoints.append((x, y, ob))
                            valid = False
                
                if not valid:
                    self.logger.warning(
                        f"Attempt {attempt} has waypoints too close to obstacles:\n"
                        f"Invalid waypoints: {invalid_waypoints}\n"
                        f"All waypoints: {waypoints}"
                    )
                    continue
                
                self.logger.info(
                    f"Successful local path generated on attempt {attempt}\n"
                    f"Generated {len(waypoints)} valid waypoints for segment {p1}->{p2}"
                )
                return waypoints
                
            except Exception as e:
                self.logger.error(
                    f"Attempt {attempt} failed with error: {str(e)}\n"
                    f"Segment: {p1}->{p2}"
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"All {max_retries} attempts failed for segment {p1}->{p2}\n"
                        f"Last error: {str(e)}"
                    )
                continue
                
        self.logger.error(f"Failed to generate valid path for segment {p1}->{p2} after {max_retries} attempts")
        raise RuntimeError(f"Failed to generate valid local path between {p1} and {p2}")