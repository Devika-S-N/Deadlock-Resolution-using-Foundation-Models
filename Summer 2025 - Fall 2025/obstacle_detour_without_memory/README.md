# Spacial Approach Agent Navigation

This project demonstrates a simple navigation strategy for an agent in a 10x10 2D occupancy grid using LIDAR and an LLM for obstacle avoidance. Here obstacle detours are done without prior knowledge of evironment and there is no memory functionality implemented.

## Folder Structure

- `environment.py`: Initializes the grid, agent, goal, and saves the environment map as a PNG.
- `lidar_obstacle.py`: Contains functions for LIDAR scanning and obstacle detection.
- `controller.py`: Controls agent movement and saves the final path as a GIF.
- `logger.py`: Logs every step to `logs/log.txt`.
- `LLM.py`: Queries Amazon Bedrock Claude model for detour waypoints.
- `main.py`: Coordinates the complete workflow.
- `logs/`: Stores the generated PNG, GIF, and log file.

## How to Run

1. Install dependencies (see requirements.txt).
2. Run `main.py`:
```bash
python3 main.py
```

All outputs will be stored in the `logs/` directory.
