import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import pandas as pd
import ast
from typing import List, Tuple

print("=== Starting Enhanced Controller ===")

# === Setup ===
folder_name = "logs_2025-07-15_21-14-56"  # Hardcoded for testing
log_dir = f"./logs/{folder_name}/"
log_file_path = os.path.join(log_dir, "planning_results.xlsx")

# Load data
df = pd.read_excel(log_file_path)
agent_start = np.array(ast.literal_eval(df.loc[0, "Agent Start"]))
goal = np.array(ast.literal_eval(df.loc[0, "Goal"]))
obstacles = [np.array(obs) for obs in ast.literal_eval(df.loc[0, "Obstacles"])]
waypoints = [np.array(p) for p in ast.literal_eval(df.loc[0, "Refined Waypoints"])]

# === Parameters ===
dt = 0.1
k = 1.0
threshold = 0.1
lidar_range = 0.5
lidar_resolution = 30  # degrees between rays
pause_frames = 10  # frames to pause at waypoints

# === Obstacle Representation ===
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

# === Simulation Data ===
rect_obstacles = [Rectangle(obs[0], obs[1], obs[2], obs[3]) for obs in obstacles]
positions = []
all_lidar_data = []

# === Simulation Function ===
def simulate_movement():
    global positions, all_lidar_data
    positions = [agent_start.copy()]
    x = agent_start.copy()


    def move_to_target(target):
        nonlocal x
        while np.linalg.norm(x - target) > threshold:
            u = -k * (x - target)
            u = np.clip(u, -1, 1)
            x = x + u * dt
            positions.append(x.copy())
            
            # LIDAR scan
            angles = np.arange(0, 360, lidar_resolution)
            lidar_results = []
            obstacle_detected = False
            for angle in angles:
                angle_rad = np.deg2rad(angle)
                end_point = x + lidar_range * np.array([np.cos(angle_rad), np.sin(angle_rad)])
                hit = any(obs.ray_intersection(x, end_point)[0] for obs in rect_obstacles)
                lidar_results.append((hit, end_point))
            all_lidar_data.append((x.copy(), lidar_results))

        # Pause at waypoint
        for _ in range(pause_frames):
            positions.append(x.copy())
            all_lidar_data.append((x.copy(), []))
    
    # Move through waypoints
    for target in waypoints:
        print(f"Moving to waypoint at {target}")
        move_to_target(target)
    
    # Move to final goal if needed
    if len(waypoints) == 0 or np.linalg.norm(waypoints[-1] - goal) > threshold:
        print("Moving to final goal")
        move_to_target(goal)

# Run simulation
print("Running simulation...")
simulate_movement()
print(f"Simulation complete. Total frames: {len(positions)}")

# === Animation Setup ===
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_title("Agent Navigation with LIDAR Scanning")

# Draw static elements
for obs in rect_obstacles:
    ax.add_patch(patches.Rectangle(
        (obs.x, obs.y), obs.width, obs.height,
        color='red', alpha=0.4, label='Obstacle'
    ))

for i, wp in enumerate(waypoints):
    ax.plot(wp[0], wp[1], 'ko', markersize=6)
    ax.text(wp[0], wp[1], f'{i+1}', ha='center', va='center')

# Initialize moving elements
agent_dot, = ax.plot([], [], 'go', markersize=10, label='Agent')
goal_dot, = ax.plot(goal[0], goal[1], 'b*', markersize=12, label='Goal')
path_line, = ax.plot([], [], 'k-', linewidth=1.5, alpha=0.7, label='Path')
lidar_lines = []
collision_marker = ax.plot([], [], 'rx', markersize=10, label='Collision')[0]
ax.legend(loc='upper right')

# === Animation Functions ===
def init():
    agent_dot.set_data([], [])
    path_line.set_data([], [])
    collision_marker.set_data([], [])
    return [agent_dot, path_line, collision_marker]

def update(frame):
    # Clear previous LIDAR lines
    for line in lidar_lines:
        line.remove()
    lidar_lines.clear()
    
    # Get current position
    pos = positions[frame]
    
    # Update agent and path
    agent_dot.set_data([pos[0]], [pos[1]])
    path_line.set_data([p[0] for p in positions[:frame+1]], 
                      [p[1] for p in positions[:frame+1]])
    
    # Update LIDAR if data exists for this frame
    collision_detected = False
    if frame < len(all_lidar_data):
        _, lidar_results = all_lidar_data[frame]
        for hit, end_point in lidar_results:
            color = 'r' if hit else 'y'
            alpha = 0.8 if hit else 0.3
            line = ax.plot([pos[0], end_point[0]], [pos[1], end_point[1]], 
                          color=color, alpha=alpha, linewidth=1)[0]
            lidar_lines.append(line)
            if hit:
                collision_detected = True
    
    # Update collision marker
    if collision_detected:
        collision_marker.set_data([pos[0]], [pos[1]])
    else:
        collision_marker.set_data([], [])
    
    return [agent_dot, path_line, collision_marker] + lidar_lines

# Create animation
print("Creating animation...")
ani = FuncAnimation(
    fig, update, frames=len(positions),
    init_func=init, blit=False, interval=50
)

# Save animation
video_path = os.path.join(log_dir, "new_object_detection_agent_path_planning.gif")
print(f"Saving animation to {video_path}...")
ani.save(video_path, writer=PillowWriter(fps=15))
print("Animation saved successfully!")

plt.show()