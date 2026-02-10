from global_planner import GlobalPlanner
from utils.visualization import visualize_path
from utils.logger import setup_logger
import os
import pandas as pd

def main():
    logger, log_dir = setup_logger()
    logger.info("=== Naive Global Path Planning Test ===")

    agent = [1, 1]
    goal = [8, 8]
    grid_size = (10, 10)
    obstacles = [(3, 3, 0.5, 1), (6, 1, 2, 1), (2, 5, 3, 0.5), (7, 2, 1, 2)]  # no obstacles involved

    global_planner = GlobalPlanner()
    waypoints = global_planner.plan_global_path(agent, goal)

    full_path = [agent] + waypoints + [goal]
    logger.info(f"Generated {len(waypoints)} waypoints")
    logger.info(f"Full Path: {full_path}")

    # Visualization
    image_path = os.path.join(log_dir, "naive_path.png")
    visualize_path(agent, goal, waypoints, obstacles, grid_size, save_path=image_path)
    logger.info(f"Path visualization saved to {image_path}")

    # Save to Excel
    df = pd.DataFrame({
        "Agent Start": [agent],
        "Goal": [goal],
        "Obstacles": [obstacles],
        "Initial Waypoints": [full_path],
        "Refined Waypoints": [full_path]  # same as initial in this case
    })
    df.to_excel(os.path.join(log_dir, "planning_results.xlsx"), index=False)
    logger.info("Path saved to Excel")

if __name__ == "__main__":
    main()
