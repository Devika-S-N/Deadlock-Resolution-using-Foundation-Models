from utils.llm_interface import LLMInterface
from utils.geometry import get_relevant_obstacles, waypoint_clear_of_obstacle, line_intersects_obstacle
import logging
import math

class LocalPlanner:
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.logger = logging.getLogger('path_planner')
        self.safety_buffer = 0.5  # Minimum distance from obstacles
        
    def _rect_to_polygon(self, obstacle):
        """Convert rectangular obstacle to polygon with safety buffer"""
        ox, oy, w, h = obstacle
        r = self.safety_buffer
        return [
            (ox - r, oy - r),  # Bottom-left
            (ox + w + r, oy - r),  # Bottom-right
            (ox + w + r, oy + h + r),  # Top-right
            (ox - r, oy + h + r),   # Top-left
            (ox - r, oy - r)  # Close the polygon
        ]

    def _find_connected_obstacles(self, obstacles):
        """Group touching/overlapping obstacles into combined polygons"""
        groups = []
        for obs in obstacles:
            matched = False
            for group in groups:
                for member in group:
                    if self._rectangles_connected(obs, member):
                        group.append(obs)
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                groups.append([obs])
        
        # Convert each group to a combined polygon
        polygons = []
        for group in groups:
            all_points = []
            for obs in group:
                all_points.extend(self._rect_to_polygon(obs)[:-1])  # Exclude closing point
            if len(all_points) >= 3:
                # Create convex hull to merge the rectangles
                polygons.append(self._create_convex_hull(all_points))
        return polygons

    def _rectangles_connected(self, rect1, rect2):
        """Check if two rectangles touch or overlap"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return not (x1 + w1 < x2 or 
                   x2 + w2 < x1 or 
                   y1 + h1 < y2 or 
                   y2 + h2 < y1)

    def _create_convex_hull(self, points):
        """Create convex hull polygon from points (simplified implementation)"""
        # This is a simplified convex hull implementation
        # For production, consider using a proper convex hull algorithm
        if len(points) <= 3:
            return points + [points[0]]  # Close the polygon
            
        # Sort by x then y
        points = sorted(points)
        # Get extreme points
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        
        # Create bounding box (simplified convex hull)
        return [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y)  # Close the polygon
        ]

    def generate_detour_prompt(self, p1, p2, obstacles):
        """Generate prompt with polygonal obstacle descriptions"""
        # Convert rectangular obstacles to polygons
        obstacle_polygons = self._find_connected_obstacles(obstacles)
        
        obstacle_descriptions = []
        for i, poly in enumerate(obstacle_polygons, 1):
            obstacle_descriptions.append(
                f"Obstacle {i} Polygon (avoid by {self.safety_buffer} units):\n"
                f"- Vertices: {poly}\n"
                f"- Safe path must go outside this shape"
            )
        
        prompt = (
            f"Generate a detour path from {p1} to {p2} around polygonal obstacles.\n"
            f"Critical Requirements:\n"
            f"1. Stay exactly {self.safety_buffer} units outside all obstacle polygons\n"
            f"2. Generate 5-7 waypoints following obstacle contours\n"
            f"3. Each segment ≤0.2 units\n"
            f"4. Only include essential turning points\n\n"
            f"Obstacle Polygons:\n"
            + "="*50 + "\n"
            + "\n".join(obstacle_descriptions) + "\n"
            + "="*50 + "\n\n"
            f"Output ONLY this format:\n"
            f"###OUTPUT_START###\n[[x1,y1],[x2,y2],...]\n###OUTPUT_END###\n"
            f"Example for going around a polygon: [[6.5,3],[7.7,3],[7.7,4.2],[8.5,3]]"
        )
        return prompt

    def validate_detour(self, waypoints, obstacles, p1, p2):
        """Strict validation of edge-following waypoints"""
        full_path = [p1] + waypoints + [p2]
        
        # Check path doesn't intersect original rectangular obstacles
        for i in range(len(full_path)-1):
            for obs in obstacles:
                if line_intersects_obstacle(full_path[i], full_path[i+1], obs):
                    return False, f"Path intersects obstacle {obs}"
        
        return True, "Validation passed"

    def plan_local_path(self, p1, p2, obstacles, max_retries=20):
        """Generate local detour with polygonal obstacle descriptions"""
        self.logger.info("\n%s", '='*100)
        self.logger.info("Generating detour between %s->%s with %d obstacles", p1, p2, len(obstacles))
        
        for attempt in range(1, max_retries + 1):
            self.logger.info("\n\n ----> Attempt %d/%d", attempt, max_retries)
            print(f"local path planning {attempt}/{max_retries}")
            
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