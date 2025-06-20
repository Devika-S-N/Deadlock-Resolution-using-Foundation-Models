from utils.llm_interface import LLMInterface
from utils.geometry import is_inside_obstacle
import logging

class GlobalPlanner:
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.logger = logging.getLogger('path_planner')
        
    def build_axis_exclusion_ranges(self, obstacles):
        """Build x and y exclusion ranges from obstacles"""
        self.logger.debug("Building axis exclusion ranges from obstacles")
        x_ranges = []
        y_ranges = []
        for ox, oy, w, h in obstacles:
            x_ranges.append((ox, ox + w))
            y_ranges.append((oy, oy + h))
        return x_ranges, y_ranges

    def generate_prompt(self, agent, goal, obstacles, grid_size, r=0.5):
        """Generate prompt for initial path planning"""
        self.logger.debug("Generating global path planning prompt")
        x_ranges, y_ranges = self.build_axis_exclusion_ranges(obstacles)
        x_rule = " or ".join([f"x in ({x1}, {x2})" for x1, x2 in x_ranges])
        y_rule = " or ".join([f"y in ({y1}, {y2})" for y1, y2 in y_ranges])
        
        prompt = (
            f"You are an AI assistant solving a 2D path planning task on a grid of size {grid_size}.\n"
            f"The agent starts at position {agent} and must reach the goal at {goal}.\n"
            f"Obstacles are defined as (x, y, width, height): {obstacles}.\n\n"
            f"Path Planning Rules:\n"
            f"1. Each waypoint's x-coordinate must NOT fall in: {x_rule}\n"
            f"2. Each waypoint's y-coordinate must NOT fall in: {y_rule}\n"
            f"3. No straight line connecting the waypoints may intersect any obstacle.\n"
            f"4. Each waypoint must be at least {r} units away from obstacle boundaries.\n"
            f"5. The Euclidean distance between adjacent waypoints must NOT exceed {r} units.\n\n"
            f"Your task:\n"
            f"Return a list of more than 8 intermediate waypoints (excluding start and goal).\n"
            f"Output strictly in this format:\n"
            f"###OUTPUT_START###\n[[x1, y1], [x2, y2], ...]\n###OUTPUT_END###\n"
        )
        self.logger.debug(f"Generated prompt:\n{prompt}")
        return prompt

    def plan_global_path(self, agent, goal, obstacles, grid_size, max_attempts=50):
        """Generate initial global path"""
        self.logger.info(f"Starting global path planning (max attempts: {max_attempts})")
        
        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"\n\n =======================================================================================================================")
            self.logger.info(f"\n----------> Attempt {attempt}/{max_attempts}")
            print(f"Global path planning attempt {attempt}/{max_attempts} ")
            
            try:
                # Generate and log prompt
                prompt = self.generate_prompt(agent, goal, obstacles, grid_size)
                #self.logger.debug(f"Attempt {attempt} Prompt:\n{prompt}")
                
                # Query LLM and log response
                response = self.llm.query_llm(prompt)
                self.logger.debug(f"Attempt {attempt} Raw Response:\n{response}")
                
                # Extract and validate waypoints
                waypoints = self.llm.extract_waypoints(response)
                self.logger.debug(f"Attempt {attempt} Extracted Waypoints: {waypoints}")
                
                if not waypoints:
                    self.logger.warning(f"Attempt {attempt} returned no waypoints")
                    continue
                    
                if len(waypoints) < 3:
                    self.logger.warning(
                        f"Attempt {attempt} returned only {len(waypoints)} waypoints "
                        f"(minimum 3 required, wanted >8)"
                    )
                    continue
                
                # Validate waypoints against obstacles
                invalid_pts = [
                    pt for pt in waypoints 
                    if is_inside_obstacle(pt[0], pt[1], obstacles)
                ]
                
                if invalid_pts:
                    self.logger.warning(
                        f"Attempt {attempt} has waypoints inside obstacles:\n"
                        f"Invalid waypoints: {invalid_pts}\n"
                        f"All waypoints: {waypoints}"
                    )
                    continue
                
                self.logger.info(
                    f"Global path planning successful on attempt {attempt}\n"
                    f"Generated {len(waypoints)} valid waypoints"
                )
                return waypoints
                
            except Exception as e:
                self.logger.error(
                    f"Attempt {attempt} failed with error: {str(e)}\n"
                    f"Agent: {agent}, Goal: {goal}"
                )
                if attempt == max_attempts:
                    self.logger.error(
                        f"All {max_attempts} attempts failed for global path planning\n"
                        f"Last error: {str(e)}"
                    )
                    raise RuntimeError(
                        "Failed to generate valid global path after multiple attempts"
                    )
                continue
                
        self.logger.error("Global path planning failed all attempts")
        raise RuntimeError("Failed to generate valid global path after multiple attempts")