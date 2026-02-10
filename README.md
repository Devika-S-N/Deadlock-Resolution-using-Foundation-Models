# Deadlock Resolution using Foundation Models  

This repository implements a **simulation-based approach for resolving deadlocks** in autonomous navigation systems using foundation models. The system generates **safe and collision-free path plans** for an agent navigating a 2D grid with static obstacles, combining high-level reasoning (VLM suggestions) with local corrections for obstacle avoidance. 

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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 # LIDAR + VLM Based Local Navigation

This repository implements a **2D mobile robot navigation system** where an agent moves toward a goal using a controller, detects obstacles using a simulated LIDAR, and resolves deadlocks by querying a **Vision‑Language Model (VLM)** for local detour waypoints.

---

## Project Structure

```

Deadlock-Resolution-using-Foundation-Models/  
│  
├── Pipeline/               # Core planning algorithms  
│   ├── main.py        
│   ├── controller.py 
│   ├── environment.py 
│   ├── warehouse.py 
│   ├── lidar_obstacle.py
│   ├── VLM.py
│   ├── memory.py
│   ├── logger.py 
│   └── logs/
│  
├── Scene_and_waypoint_acccuracy_files             
├── Summer 2025 - Fall 2025            
└── README.md

```

---

## Requirements

### Python Version

* Python **3.9+** recommended

### Python Packages

Install the following packages:

```
pip install numpy matplotlib shapely scikit-learn openai
```

Optional (for video recording):

* `ffmpeg` must be installed on your system

Linux:

```
sudo apt install ffmpeg
```

macOS:

```
brew install ffmpeg
```

---

## Environment Setup

1. **Clone the repository**

```
git clone <repo-url>
cd <repo-name>
```

2. **Create a virtual environment (optional but recommended)**

```
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```
pip install -r requirements.txt
```

(If `requirements.txt` is not present, install manually using the list above.)

4. **Set OpenAI API Key**

This project uses OpenAI VLMs.

```
export OPENAI_API_KEY="your_api_key_here"
```

---

## How to Run

### Main Entry Point

Run the full navigation pipeline. Navigate to the Pipeline folder and proceed:

```
python main.py
```

This will:

* Generate a random warehouse environment
* Start the robot at the agent location
* Perform LIDAR scans
* Query the VLM when obstacles are detected
* Log results and save videos/plots

Outputs are saved inside the `logs/` folder.

---

## File‑by‑File Explanation

### `main.py`

* Entry point of the project
* Creates a `NavigationController` and starts navigation

---

### `controller.py`

* **Core navigation logic**
* Moves the agent toward the goal
* Runs LIDAR scans every step
* Detects obstacles
* Calls the VLM for detour waypoints
* Validates waypoints against real obstacles
* Generates navigation animation and logs

---

### `environment.py`

* Interface between `warehouse.py` and the controller
* Converts grid‑based warehouse output into:

  * Agent position
  * Goal position
  * Obstacle polygons
* Handles coordinate conversion
* Saves a debug image of the environment

---

### `warehouse.py`

* Procedural **warehouse / room environment generator**
* Creates walls, rooms, and obstacles
* Saves environment as JSON + PNG
* Used only during environment creation

---

### `lidar_obstacle.py`

* Simulates a **2D LIDAR sensor**
* Casts radial rays around the agent
* Returns hit points where rays intersect obstacles
* Used for obstacle detection and VLM input

---

### `VLM.py`

* Handles **Vision‑Language Model interaction**
* Builds structured prompts using:

  * Agent position
  * Goal position
  * LIDAR hit points
  * Previous waypoints
* Sends image + text to OpenAI model
* Parses JSON waypoint output

---

### `memory.py`

* Maintains a **live heatmap** of:

  * Agent trajectory
  * LIDAR hit density
* Performs obstacle clustering
* Saves:

  * Heatmap video
  * Clean snapshots for VLM queries

---

### `logger.py`

* Simple thread‑safe logger
* Writes ordered logs with timestamps
* Used across all modules

---

### `detour_planner.py`

* Legacy / experimental local planner
* Includes PCA‑based wall following and RRT
* Not used in the main pipeline by default

---

## Logs and Outputs

Inside `logs/run_<timestamp>/` you will find:

* `log_<timestamp>.txt` → detailed execution logs
* `navigation_<timestamp>.mp4` → navigation animation
* `heatmap_<timestamp>.mp4` → LIDAR + path heatmap
* `vlm_snapshot_*.png` → images sent to the VLM

---

## Notes

* Grid size is fixed to **15 × 15** by default
* LIDAR range is **1.0 unit**
* VLM retries automatically if bad waypoints are returned
* Designed for **research and debugging**, not real‑time use

---

## Troubleshooting

* If VLM fails: check `OPENAI_API_KEY`
* If video not saved: ensure `ffmpeg` is installed
* If plots freeze: close Matplotlib windows manually

---

## Author / Project Context

This codebase is designed for **research on deadlock resolution in robot navigation using foundation models**, combining classical control, perception, and VLM‑based reasoning.

