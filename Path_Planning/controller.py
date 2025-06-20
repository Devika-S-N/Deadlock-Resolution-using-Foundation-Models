# Final re-run to save the animation after repeated disconnections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import time
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
import os
import pandas as pd
import ast  

# Load the Excel file
folder_name = str(input("Enter the folder name:"))
log_dir = f"/home/devika/Desktop/MORE_project/Deadlock_Resolution/Deadlock-Resolution-using-Foundation-Models/Path_Planning/logs/logs/{folder_name}/"
log_file_path = os.path.join(log_dir, "test_results.xlsx")
df = pd.read_excel(log_file_path)


# Extract and convert string data to Python objects
agent_start = np.array(ast.literal_eval(df.loc[0, "Agent"]))
goal = np.array(ast.literal_eval(df.loc[0, "Goal"]))
obstacles = ast.literal_eval(df.loc[0, "Obstacles"])
waypoints = [np.array(p) for p in ast.literal_eval(df.loc[0, "Refined Waypoints"])]
dt = 0.1
k = 1.0
threshold = 0.1

# === Controller Function ===
def controller(x, target):
    u = -k * (x - target)
    u = np.clip(u, -1, 1)
    return u

# === Simulate Agent Movement ===
positions = [agent_start.copy()]
x = agent_start.copy()

for target in waypoints:
    while np.linalg.norm(x - target) > threshold:
        u = controller(x, target)
        x = x + u * dt
        positions.append(x.copy())
    for _ in range(10):
        positions.append(x.copy())

positions = np.array(positions)

# === Setup Plot ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Agent Path Planning")

# Draw obstacles
for obs in obstacles:
    ax.add_patch(patches.Rectangle((obs[0], obs[1]), obs[2], obs[3], color='red', alpha=0.4))

agent_dot, = ax.plot([], [], 'go', label="Agent")
goal_dot, = ax.plot(goal[0], goal[1], 'bo', label="Goal")
path_line, = ax.plot([], [], 'k--', linewidth=1, label="Path")

# Waypoints
for wp in waypoints:
    ax.plot(wp[0], wp[1], 'ko', markersize=3)

ax.legend()

def init():
    agent_dot.set_data([], [])
    path_line.set_data([], [])
    return agent_dot, path_line

def update(frame):
    agent_dot.set_data([positions[frame][0]], [positions[frame][1]])
    path_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
    return agent_dot, path_line

ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, interval=100)

# Save animation
video_path = os.path.join(log_dir, "agent_path_planning.gif")
ani.save(video_path, writer=PillowWriter(fps=10))

video_path
