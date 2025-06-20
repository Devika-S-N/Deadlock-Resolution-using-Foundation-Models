# Deadlock Resolution using Foundation Models

This repository presents a simulation-based approach for resolving deadlocks in autonomous navigation systems using foundation models. It focuses on generating safe and efficient path plans for agents navigating in a 2D grid with static obstacles.

 ## Objective

    Use LLM-generated paths to navigate an agent from a start to a goal location in a 2D environment.

    Avoid static obstacles using a combination of global planning (LLM suggestions) and local correction (in case of obstacle interference).

## Folder Structure

Deadlock-Resolution-using-Foundation-Models/
│
├── path_planning/
│ ├── global_planner.py # Global planner using high-level reasoning
│ ├── local_planner.py # Local planner for obstacle avoidance
│ ├── main.py # Script to test global path planning
│ ├── test.py # Script to test local path planning
│ └── controller.py # Script that defines a P controller to navigate the waypoints
│
├── logs/ # Folder where logs and visual outputs are saved
│
└── (Other folders and scripts related to Single agent planning using LLAMA model)


## How to Run

### Global Path Planning
To test the global path planner:
python3 path_planning/main.py


###Local Path Planning
To test the local planner:
python path_planning/test.py

### Requirements

    Python 3.x

    boto3

    matplotlib

    pandas

    shapely


