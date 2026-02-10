import boto3
import json
import re
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import pandas as pd

# AWS Setup
os.environ['AWS_PROFILE'] = "MyProfile1"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
bedrock = boto3.client("bedrock-runtime")

# Timestamped log folder
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
log_dir = f"logs/logs_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "log.txt")

# Input
agent = [1, 1]
goal = [5, 9]
obstacles = [(3, 3, 0.5, 1), (6, 1, 2, 1), (2, 5, 3, 0.5), (7, 2, 1, 2)]
grid_size = (10, 10)

def log_entry(content):
    with open(log_file_path, "a") as f:
        f.write(content + "\n")

def is_inside_obstacle(x, y, obstacles):
    for ox, oy, w, h in obstacles:
        if ox <= x <= ox + w and oy <= y <= oy + h:
            return True
    return False

def extract_waypoints(text):
    import ast
    text = text.replace("`", "")

    match = re.search(r"###OUTPUT_START###(.*?)###OUTPUT_END###", text, re.DOTALL)
    if match:
        snippet = match.group(1).strip()
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    list_match = re.search(r"\[\s*\[[^\[\]]+\](?:\s*,\s*\[[^\[\]]+\])*\s*\]", text)
    if list_match:
        snippet = list_match.group(0)
        try:
            return ast.literal_eval(snippet)
        except Exception:
            return None

    return None

def build_axis_exclusion_ranges(obstacles):
    x_ranges = []
    y_ranges = []
    for ox, oy, w, h in obstacles:
        x_ranges.append((ox, ox + w))
        y_ranges.append((oy, oy + h))
    return x_ranges, y_ranges

def generate_prompt(agent, goal, obstacles, r=0.5):
    x_ranges, y_ranges = build_axis_exclusion_ranges(obstacles)
    x_rule = " or ".join([f"x in ({x1}, {x2})" for x1, x2 in x_ranges])
    y_rule = " or ".join([f"y in ({y1}, {y2})" for y1, y2 in y_ranges])

    return (
        f"You are an AI assistant solving a 2D path planning task on a grid of size {grid_size}.\n"
        f"The agent starts at position {agent} and must reach the goal at {goal}.\n"
        f"Obstacles are defined as (x, y, width, height): {obstacles}.\n\n"
        f"Path Planning Rules:\n"
        f"1. Each waypoint's x-coordinate must NOT fall in: {x_rule}\n"
        f"2. Each waypoint's y-coordinate must NOT fall in: {y_rule}\n"
        f"3. No straight line connecting the waypoints (including start -> first waypoint and last waypoint -> goal) may intersect any obstacle.\n"
        f"4. Each waypoint must be at least {r} units away from the boundary of any obstacle.\n"
        f"5. The Euclidean distance between any two adjacent waypoints (or start/goal) must NOT exceed {r} units.\n\n"
        f"Your task:\n"
        f"Return a list of more than 8 intermediate waypoints (excluding the start and goal).\n"
        f"Output strictly in this format:\n"
        f"###OUTPUT_START###\n[[x1, y1], [x2, y2], [x3, y3], ...]\n###OUTPUT_END###\n\n"
        f"Do NOT include explanations, code, or anything outside the delimiters."
    )

def visualize_path(agent, goal, waypoints, obstacles, grid_size, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_xticks(range(grid_size[0] + 1))
    ax.set_yticks(range(grid_size[1] + 1))
    ax.grid(True)
    ax.set_aspect('equal')

    for ox, oy, w, h in obstacles:
        rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='red', facecolor='red', alpha=0.5)
        ax.add_patch(rect)

    ax.plot(agent[0], agent[1], 'go', label="Agent", markersize=10)
    ax.plot(goal[0], goal[1], 'bo', label="Goal", markersize=10)

    if waypoints:
        wp_x, wp_y = zip(*waypoints)
        ax.plot([agent[0]] + list(wp_x) + [goal[0]],
                [agent[1]] + list(wp_y) + [goal[1]],
                'k--', label="Path")
        ax.plot(list(wp_x), list(wp_y), 'ko', markersize=6, label="Waypoints")

    ax.legend(loc='upper right')
    plt.title("Agent Path Planning")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

# Main loop
attempt = 0
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

while True:
    attempt += 1
    r = 0.5
    prompt_text = generate_prompt(agent, goal, obstacles, r=r)

    print(f"\nAttempt: {attempt}")
    log_entry(f"\n=== Attempt {attempt} ===\n{'*' * 130}")
    log_entry("Prompt:\n" + prompt_text + "\n" + "*" * 130)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 150,
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
            }
        ]
    })

    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response["body"].read())
        response_text = response_body["content"][0]["text"]

        log_entry("Response:\n" + response_text + "\n" + "*" * 130)

        waypoints = extract_waypoints(response_text)
        print(f"**************Waypoints = {waypoints}")

        if not waypoints or len(waypoints) < 3:
            log_entry("Invalid response: Less than 3 waypoints. Retrying...\n" + "*" * 130)
            continue

        invalid_pts = [pt for pt in waypoints if is_inside_obstacle(pt[0], pt[1], obstacles)]
        if invalid_pts:
            log_entry(f"Invalid waypoints in obstacles: {invalid_pts}. Retrying...\n" + "*" * 130)
            continue

        log_entry(f"Valid waypoints obtained: {waypoints}\n" + "*" * 130)
        image_path = os.path.join(log_dir, "final_path.png")
        visualize_path(agent, goal, waypoints, obstacles, grid_size, save_path=image_path)
        log_entry(f"Saved visualization to: {image_path}")

        # Save details to Excel
        excel_data = {
            "Agent Start": [agent],
            "Goal": [goal],
            "Obstacles": [obstacles],
            "Final Waypoints": [waypoints]
        }
        df = pd.DataFrame(excel_data)
        excel_path = os.path.join(log_dir, "path_data.xlsx")
        df.to_excel(excel_path, index=False)
        log_entry(f"Saved data to Excel: {excel_path}")

        break

    except Exception as e:
        log_entry(f"Error occurred: {str(e)}\nRetrying...\n" + "*" * 130)
        time.sleep(1)
