from utils.llm_interface import LLMInterface
from utils.geometry import is_inside_obstacle
import logging
import math

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
            x_ranges.append((ox - 0.5, ox + w + 0.5))  # Added buffer
            y_ranges.append((oy - 0.5, oy + h + 0.5))  # Added buffer
        return x_ranges, y_ranges

    def generate_prompt(self, agent, goal, obstacles, grid_size):
        """Generate prompt for initial path planning"""
        self.logger.debug("Generating global path planning prompt")
        x_ranges, y_ranges = self.build_axis_exclusion_ranges(obstacles)
        x_rule = " or ".join([f"x in ({x1:.1f}, {x2:.1f})" for x1, x2 in x_ranges])
        y_rule = " or ".join([f"y in ({y1:.1f}, {y2:.1f})" for y1, y2 in y_ranges])
        
        prompt = (
            f"You are an AI assistant solving a 2D path planning task on a grid of size {grid_size}.\n"
            f"The agent starts at position {agent} and must reach the goal at {goal}.\n"
            f"Obstacles are defined as (x, y, width, height): {obstacles}.\n\n"
            f"Path Planning Rules:\n"
            f"1. Generate at least 15 intermediate waypoints (excluding start and goal).\n"
            f"2. Each waypoint should maintain at least 0.5 units distance from all obstacles.\n"
            f"3. The distance between adjacent waypoints must not exceed 0.2 units.\n"
            f"4. The path should generally avoid these regions:\n"
            f"   - x-coordinates to avoid: {x_rule}\n"
            f"   - y-coordinates to avoid: {y_rule}\n\n"
            f"Output strictly in this format:\n"
            f"###OUTPUT_START###\n[[x1, y1], [x2, y2], ...]\n###OUTPUT_END###\n"
            f"Do NOT include explanations, code, or anything outside the delimiters."
            f"Note: If a few waypoints accidentally fall in obstacle regions, we can remove them later."
        )
        self.logger.debug(f"Generated prompt:\n{prompt}")
        return prompt

    def validate_waypoints(self, waypoints, obstacles):
        """Validate waypoints with new relaxed criteria"""
        if len(waypoints) < 8:
            return False, "Too few waypoints"
                
        # Remove waypoints inside obstacles (with buffer)
        valid_waypoints = [
            pt for pt in waypoints 
            if not is_inside_obstacle(pt[0], pt[1], obstacles)
        ]
        
        # Allow up to 2 invalid waypoints to be removed
        if len(waypoints) - len(valid_waypoints) > 2:
            return False, f"Too many ({len(waypoints)-len(valid_waypoints)}) waypoints in obstacles"
            
        return True, valid_waypoints if valid_waypoints else waypoints

    def plan_global_path(self, agent, goal, obstacles, grid_size, max_attempts=50):
        """Generate initial global path with relaxed constraints"""
        self.logger.info(f"Starting global path planning (max attempts: {max_attempts})")
        
        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"\n\n{'='*100}")
            self.logger.info(f"\n----------> Attempt {attempt}/{max_attempts}")
            print(f"Global path planning attempt {attempt}/{max_attempts}")
            
            try:
                prompt = self.generate_prompt(agent, goal, obstacles, grid_size)
                response = self.llm.query_llm(prompt)
                self.logger.debug(f"Attempt {attempt} Raw Response:\n{response}")
                
                waypoints = self.llm.extract_waypoints(response)
                self.logger.debug(f"Attempt {attempt} Extracted Waypoints: {waypoints}")
                
                if not waypoints:
                    self.logger.warning(f"Attempt {attempt} returned no waypoints")
                    continue
                    
                # Validate with relaxed criteria
                is_valid, result = self.validate_waypoints(waypoints, obstacles)
                
                if not is_valid:
                    self.logger.warning(f"Attempt {attempt} validation failed: {result}")
                    continue
                    
                final_waypoints = result if isinstance(result, list) else waypoints
                
                self.logger.info(
                    f"Global path planning successful on attempt {attempt}\n"
                    f"Initial waypoints: {len(waypoints)}\n"
                    f"Final waypoints after cleaning: {len(final_waypoints)}\n"
                    f"Waypoints: {final_waypoints}"
                )
                return final_waypoints
                
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