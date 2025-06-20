# Deadlock Resolution using Foundation Models  

This repository implements a **simulation-based approach for resolving deadlocks** in autonomous navigation systems using foundation models (e.g., LLMs). The system generates **safe and collision-free path plans** for agents navigating a 2D grid with static obstacles, combining high-level reasoning (LLM suggestions) with local corrections for obstacle avoidance.  

## Key Features  
- **Global Path Planning**: Uses LLM-generated waypoints for high-level navigation.  
- **Local Obstacle Avoidance**: Corrects paths dynamically to avoid collisions.  
- **Simulation & Visualization**: Logs and visualizes agent trajectories, obstacles, and deadlock resolutions.  

## Quick Start  

### Prerequisites  
- **Python 3.8+**  
- Dependencies:  
  ```bash
  pip install boto3 matplotlib pandas shapely

## Running the Simulation

-- Global Path Planning (LLM-generated waypoints):

    python3 path_planning/main.py


-- Local Path Planning (Obstacle avoidance):

    python3 path_planning/test.py

## Repository Structure
Deadlock-Resolution-using-Foundation-Models/  
│  
├── path_planning/               # Core planning algorithms  
│   ├── global_planner.py        # High-level LLM-based path generation  
│   ├── local_planner.py         # Dynamic obstacle avoidance  
│   ├── main.py                  # Test global planning  
│   ├── test.py                  # Test local planning  
│   └── controller.py            # P-controller for waypoint navigation  
│  
├── logs/                        # Simulation logs & visualizations (ignored by Git)  
│  
└── (Other LLM-related scripts)  # Single-agent planning with LLAMA  

## How It Works
### Global Planner:
 --Queries an LLM for high-level waypoints from start to goal.
 --Outputs a coarse path (may intersect obstacles).

### Local Planner:
 --Adjusts the global path in real-time to avoid static obstacles.
 --Uses a P-controller (controller.py) for smooth navigation.
