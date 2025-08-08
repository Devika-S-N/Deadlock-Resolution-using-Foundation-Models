
## second_controller.py (updated)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import pandas as pd
import ast
from typing import List, Tuple
from local_planner import LocalPlanner
from test_global_planner import GlobalPlannerWrapper

class Rectangle:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vertices = self._compute_vertices()

    def _compute_vertices(self) -> List[Tuple[float, float]]:
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]

    def ray_intersection(self, ray_start: np.ndarray, ray_end: np.ndarray) -> Tuple[bool, np.ndarray]:
        x1, y1 = ray_start
        x2, y2 = ray_end
        vertices = self.vertices

        min_alpha = float('inf')
        intersection = None

        for i in range(4):
            x3, y3 = vertices[i]
            x4, y4 = vertices[(i + 1) % 4]

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue

            alpha = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            beta = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

            if 0 <= alpha <= 1 and 0 <= beta <= 1:
                if alpha < min_alpha:
                    min_alpha = alpha
                    intersection = np.array([
                        x1 + alpha * (x2 - x1),
                        y1 + alpha * (y2 - y1)
                    ])

        return (intersection is not None, intersection)

class NavigationController:
    def __init__(self):
        self.global_planner = GlobalPlannerWrapper()
        self.local_planner = LocalPlanner(aws_profile="MyProfile1", aws_region="us-east-1")

        folder_name = "logs_2025-08-02_14-52-14"
        self.log_dir = f"./logs/{folder_name}/"
        df = pd.read_excel(os.path.join(self.log_dir, "planning_results.xlsx"))

        self.agent_start = np.array(ast.literal_eval(df.loc[0, "Agent Start"]))
        self.goal = np.array(ast.literal_eval(df.loc[0, "Goal"]))
        raw_obstacles = ast.literal_eval(df.loc[0, "Obstacles"])

        self.obstacles = [np.array(obs) for obs in raw_obstacles]
        self.rect_obstacles = [Rectangle(obs[0], obs[1], obs[2], obs[3]) for obs in self.obstacles]

        self.waypoints = self.global_planner.plan_path(tuple(self.agent_start), tuple(self.goal))

        self.dt = 0.1
        self.k = 1.0
        self.threshold = 0.1
        self.lidar_range = 0.5
        self.lidar_resolution = 30
        self.safety_buffer = 0.3

        self.positions = [self.agent_start.copy()]
        self.all_lidar_data = []
        self.current_target_index = 0
        self.processed_obstacles = set()

    def run_simulation(self):
        log_file = open(os.path.join(self.log_dir, "navigation_log_3.txt"), "w")
        try:
            x = self.agent_start.copy()
            while True:
                if self.current_target_index >= len(self.waypoints):
                    if np.linalg.norm(x - self.goal) <= self.threshold:
                        break
                    self._replan_global_path(x)
                    continue
                target = self.waypoints[self.current_target_index]
                self._log(f"Moving to waypoint {self.current_target_index+1}/{len(self.waypoints)}: {target}", log_file)
                x = self._move_to_target(x, target, log_file)
                if np.linalg.norm(x - target) <= self.threshold:
                    self.current_target_index += 1
        finally:
            log_file.close()
            self._create_animation()

    def _move_to_target(self, x: np.ndarray, target: np.ndarray, log_file) -> np.ndarray:
        while np.linalg.norm(x - target) > self.threshold:
            u = -self.k * (x - target)
            u = np.clip(u, -1, 1)
            x = x + u * self.dt
            self.positions.append(x.copy())

            lidar_results = []
            current_obstacle = None
            collision_point = None

            for angle in np.arange(0, 360, self.lidar_resolution):
                angle_rad = np.deg2rad(angle)
                end_point = x + self.lidar_range * np.array([np.cos(angle_rad), np.sin(angle_rad)])
                for obs in self.rect_obstacles:
                    hit, point = obs.ray_intersection(x, end_point)
                    if hit:
                        obs_key = (
                            round(obs.x, 1), round(obs.y, 1),
                            round(obs.width, 1), round(obs.height, 1),
                            round(x[0], 1), round(x[1], 1)
                        )
                        if obs_key not in self.processed_obstacles:
                            current_obstacle = obs
                            collision_point = point
                        lidar_results.append((True, point))
                        break
                    else:
                        lidar_results.append((False, end_point))

            self.all_lidar_data.append((x.copy(), lidar_results))

            if current_obstacle and collision_point is not None:
                self._handle_obstacle(x, target, current_obstacle, collision_point, log_file)
                return x

        return x

    def _handle_obstacle(self, x: np.ndarray, target: np.ndarray, obstacle, collision_point: np.ndarray, log_file):
        obs_coords = (obstacle.x, obstacle.y, obstacle.width, obstacle.height)
        self._log(f"Obstacle detected at {obs_coords}, collision at {collision_point}", log_file)

        detour = self.local_planner.generate_detour(
            current_pos=tuple(x),
            collision_point=tuple(collision_point),
            next_waypoint=tuple(target),
            goal=tuple(self.goal),
            obstacle=obs_coords,
            buffer=self.safety_buffer
        )

        if detour:
            self._log(f"Generated detour: {detour}", log_file)
            self.processed_obstacles.add(obs_coords)
            self.waypoints = self.waypoints[:self.current_target_index] + [np.array(wp) for wp in detour]
            self.current_target_index = 0
        else:
            self._log("No valid detour found - stopping", log_file)
            self.waypoints = [x.copy()]

    def _replan_global_path(self, current_pos: np.ndarray):
        self._log("Replanning global path from current position", None)
        self.waypoints = self.global_planner.plan_path(tuple(current_pos), tuple(self.goal))
        self.current_target_index = 0

    def _log(self, message: str, log_file):
        print(message)
        if log_file:
            log_file.write(message + "\n")

    def _create_animation(self):
       
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_title("Agent Navigation with LIDAR Scanning")

        # Draw static elements
        for obs in self.rect_obstacles:
            ax.add_patch(patches.Rectangle(
                (obs.x, obs.y), obs.width, obs.height,
                color='red', alpha=0.4, label='Obstacle'
            ))
            
        # Initialize moving elements
        agent_dot, = ax.plot([], [], 'go', markersize=10, label='Agent')
        goal_dot, = ax.plot(self.goal[0], self.goal[1], 'b*', markersize=12, label='Goal')
        path_line, = ax.plot([], [], 'k-', linewidth=1.5, alpha=0.7, label='Path')
        lidar_lines = []
        collision_markers = []
        
        def init():
            agent_dot.set_data([], [])
            path_line.set_data([], [])
            return [agent_dot, path_line]
        
        def update(frame):
            # Clear previous LIDAR lines and markers
            for line in lidar_lines:
                line.remove()
            for marker in collision_markers:
                marker.remove()
            lidar_lines.clear()
            collision_markers.clear()
            
            # Get current position
            pos = self.positions[frame]
            
            # Update agent and path
            agent_dot.set_data([pos[0]], [pos[1]])
            path_line.set_data([p[0] for p in self.positions[:frame+1]], 
                                [p[1] for p in self.positions[:frame+1]])
            
            # Update LIDAR if data exists
            if frame < len(self.all_lidar_data):
                _, lidar_results = self.all_lidar_data[frame]
                for hit, point in lidar_results:
                    if hit:
                        # Draw LIDAR ray to collision point
                        line = ax.plot([pos[0], point[0]], [pos[1], point[1]], 
                                        color='r', alpha=0.8, linewidth=1)[0]
                        lidar_lines.append(line)
                        # Mark collision point
                        marker = ax.plot(point[0], point[1], 'rx', markersize=8)[0]
                        collision_markers.append(marker)
                    else:
                        # Draw regular LIDAR ray
                        line = ax.plot([pos[0], point[0]], [pos[1], point[1]], 
                                        color='y', alpha=0.3, linewidth=1)[0]
                        lidar_lines.append(line)
            
            return [agent_dot, path_line] + lidar_lines + collision_markers
        
        ani = FuncAnimation(
            fig, update, frames=len(self.positions),
            init_func=init, blit=False, interval=50
        )
        
        video_path = os.path.join(self.log_dir, "navigation_animation_3.gif")
        ani.save(video_path, writer=PillowWriter(fps=15))
        plt.close()

if __name__ == "__main__":
    controller = NavigationController()
    print("=== Starting Navigation ===")
    controller.run_simulation()
    print("=== Navigation Completed ===")