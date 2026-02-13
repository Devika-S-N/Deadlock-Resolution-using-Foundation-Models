# VLM.py
# A* planner using REAL obstacles (not LIDAR-built ones)
# This ensures waypoints pass controller validation

import numpy as np
import heapq
from typing import List, Tuple, Optional
from shapely.geometry import Point, LineString, Polygon


class RealObstacleAStarPlanner:
    """
    A* planner that uses REAL obstacles from the environment.
    Ensures generated waypoints will pass controller validation.
    """
    
    def __init__(self, resolution=0.05, obstacle_margin=0.4, local_region_size=4.0):
        self.resolution = resolution
        self.obstacle_margin = obstacle_margin
        self.local_region_size = local_region_size
        self.last_waypoints = None
        self.real_obstacles = None  # Will be set by controller
    
    def set_real_obstacles(self, obstacles):
        """Controller calls this to provide real obstacles"""
        self.real_obstacles = obstacles
    
    def plan_local_detour(self, agent, goal, lidar_hits):
        """
        Plan waypoints using REAL obstacles (not LIDAR).
        """
        
        if self.real_obstacles is None:
            print("ERROR: Real obstacles not set!")
            return None
        
        # Calculate local goal
        local_goal = self._calculate_local_goal(agent, goal, lidar_hits)
        
        # Run A* using REAL obstacles
        path = self._astar_with_real_obstacles(agent, local_goal)
        
        if path is None or len(path) < 2:
            return None
        
        # Sample 4 waypoints
        waypoints = self._sample_waypoints(path, num_waypoints=4)
        
        # Smooth
        waypoints = self._smooth_waypoints(waypoints)
        
        self.last_waypoints = waypoints
        return waypoints
    
    def _calculate_local_goal(self, agent, final_goal, lidar_hits):
        """Calculate local goal with momentum"""
        agent_arr = np.array(agent)
        goal_arr = np.array(final_goal)
        
        direction = goal_arr - agent_arr
        dist_to_goal = np.linalg.norm(direction)
        
        if dist_to_goal < 0.01:
            return final_goal
        
        direction_norm = direction / dist_to_goal
        
        # Use momentum if available
        if self.last_waypoints is not None and len(self.last_waypoints) > 0:
            last_wp = np.array(self.last_waypoints[-1])
            prev_direction = last_wp - agent_arr
            prev_dist = np.linalg.norm(prev_direction)
            
            if prev_dist > 0.1:
                prev_direction_norm = prev_direction / prev_dist
                # 60% previous + 40% goal
                blended = 0.6 * prev_direction_norm + 0.4 * direction_norm
                blended = blended / np.linalg.norm(blended)
                
                local_distance = min(4.0, dist_to_goal)
                local_goal = agent_arr + blended * local_distance
                return local_goal.tolist()
        
        # No momentum - straight to goal
        local_distance = min(4.0, dist_to_goal)
        local_goal = agent_arr + direction_norm * local_distance
        return local_goal.tolist()
    
    def _astar_with_real_obstacles(self, start, goal):
        """A* using REAL obstacles from environment"""
        
        agent_arr = np.array(start)
        x_min = agent_arr[0] - self.local_region_size
        x_max = agent_arr[0] + self.local_region_size
        y_min = agent_arr[1] - self.local_region_size
        y_max = agent_arr[1] + self.local_region_size
        
        grid_width = int((x_max - x_min) / self.resolution)
        grid_height = int((y_max - y_min) / self.resolution)
        
        def world_to_grid(pos):
            gx = int((pos[0] - x_min) / self.resolution)
            gy = int((pos[1] - y_min) / self.resolution)
            gx = max(0, min(gx, grid_width - 1))
            gy = max(0, min(gy, grid_height - 1))
            return (gx, gy)
        
        def grid_to_world(node):
            x = x_min + node[0] * self.resolution
            y = y_min + node[1] * self.resolution
            return [x, y]
        
        def heuristic(n1, n2):
            return np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)
        
        def is_valid(pos):
            """Check against REAL obstacles (same as controller validation)"""
            point = Point(pos[0], pos[1])
            
            # Check real obstacles
            for obs in self.real_obstacles:
                if hasattr(obs, 'contains'):  # Polygon
                    if obs.contains(point):
                        return False
                    if obs.distance(point) < self.obstacle_margin:
                        return False
                elif hasattr(obs, 'distance'):  # LineString
                    if obs.distance(point) < self.obstacle_margin:
                        return False
                else:  # List of points (polygon)
                    # Convert to Polygon
                    poly = Polygon(obs)
                    if poly.contains(point):
                        return False
                    if poly.distance(point) < self.obstacle_margin:
                        return False
            
            return True
        
        start_node = world_to_grid(start)
        goal_node = world_to_grid(goal)
        
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: heuristic(start_node, goal_node)}
        closed_set = set()
        
        iterations = 0
        max_iterations = 10000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if heuristic(current, goal_node) < 3:
                # Found path
                path = [grid_to_world(current)]
                while current in came_from:
                    current = came_from[current]
                    path.append(grid_to_world(current))
                path.reverse()
                return path
            
            closed_set.add(current)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    if not (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height):
                        continue
                    
                    if neighbor in closed_set:
                        continue
                    
                    neighbor_world = grid_to_world(neighbor)
                    if not is_valid(neighbor_world):
                        continue
                    
                    tentative_g = g_score[current] + heuristic(current, neighbor)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal_node)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _sample_waypoints(self, path, num_waypoints=4):
        """Sample exactly 4 waypoints"""
        if len(path) <= num_waypoints:
            # Pad with duplicates if needed
            while len(path) < num_waypoints:
                path.append(path[-1])
            return path[:num_waypoints]
        
        distances = [0]
        for i in range(1, len(path)):
            dist = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        if total_distance < 0.01:
            return path[:num_waypoints]
        
        waypoints = []
        for i in range(1, num_waypoints + 1):
            target_distance = (i / (num_waypoints + 1)) * total_distance
            
            for j in range(len(distances) - 1):
                if distances[j] <= target_distance <= distances[j+1]:
                    t = (target_distance - distances[j]) / (distances[j+1] - distances[j] + 1e-9)
                    wp = [
                        path[j][0] + t * (path[j+1][0] - path[j][0]),
                        path[j][1] + t * (path[j+1][1] - path[j][1])
                    ]
                    waypoints.append(wp)
                    break
        
        while len(waypoints) < num_waypoints:
            waypoints.append(path[-1])
        
        return waypoints[:num_waypoints]
    
    def _smooth_waypoints(self, waypoints):
        """Smooth waypoints"""
        if len(waypoints) < 3:
            return waypoints
        
        smoothed = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            smooth_wp = [
                (waypoints[i-1][0] + waypoints[i][0] + waypoints[i+1][0]) / 3,
                (waypoints[i-1][1] + waypoints[i][1] + waypoints[i+1][1]) / 3
            ]
            smoothed.append(smooth_wp)
        smoothed.append(waypoints[-1])
        
        return smoothed


# Global planner
_planner = None

def _get_planner():
    global _planner
    if _planner is None:
        _planner = RealObstacleAStarPlanner(
            resolution=0.05,
            obstacle_margin=0.4,
            local_region_size=4.0
        )
    return _planner


def set_real_obstacles(obstacles):
    """Controller calls this to provide real obstacles"""
    planner = _get_planner()
    planner.set_real_obstacles(obstacles)


def query_detour_waypoints(
    image_path: str,
    agent: Tuple[float, float],
    goal: Tuple[float, float],
    lidar_hits: List[Tuple[float, float]],
    yellow_waypoints: List[Tuple[float, float]] = None,
    model: str = None,
    retry_note: Optional[str] = None,
    logger = None,
) -> List[Tuple[float, float]]:
    """
    A* planner using REAL obstacles.
    """
    
    if logger:
        logger.log("A* with real obstacles: Planning detour")
    
    planner = _get_planner()
    waypoints = planner.plan_local_detour(agent, goal, lidar_hits)
    
    if waypoints:
        waypoints_tuples = [tuple(wp) for wp in waypoints]
        if logger:
            logger.log(f"A* generated {len(waypoints_tuples)} waypoints")
            for i, wp in enumerate(waypoints_tuples, 1):
                logger.log(f"  WP{i}: [{wp[0]:.3f}, {wp[1]:.3f}]")
        return waypoints_tuples
    else:
        if logger:
            logger.log("A* found no path")
        return None