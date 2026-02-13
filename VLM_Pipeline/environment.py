# environment.py
import json
import numpy as np
from warehouse import generate_env  # this is your Code-2 file: warehouse.py
from shapely.geometry import LineString


def walls_from_warehouse(vwall, hwall, N):
    walls = []

    # vertical walls
    for col in range(N + 1):
        for r in range(N):
            if vwall[col][r]:
                y0 = N - (r + 1)
                y1 = N - r
                walls.append(LineString([(col, y0), (col, y1)]))

    # horizontal walls
    for row in range(N + 1):
        for c in range(N):
            if hwall[row][c]:
                y = N - row
                walls.append(LineString([(c, y), (c + 1, y)]))

    return walls

import matplotlib.pyplot as plt

def debug_render_environment(obstacles, agent, goal, N, out_path):
    fig, ax = plt.subplots(figsize=(8,8))

    for obs in obstacles:
        if hasattr(obs, "xy"):  # LineString
            x, y = obs.xy
            ax.plot(x, y, "k", linewidth=2)
        else:  # polygon
            xs, ys = zip(*(obs + [obs[0]]))
            ax.plot(xs, ys, "k", linewidth=2)
            # Fill obstacle polygons
            ax.fill(xs, ys, color='gray', alpha=0.5)

    ax.scatter(agent[0], agent[1], c="green", s=150, label="agent", zorder=10, edgecolors='darkgreen', linewidths=2)
    ax.scatter(goal[0], goal[1], c="blue", s=150, label="goal", zorder=10, edgecolors='darkblue', linewidths=2)

    # Draw ALL integer grid lines
    for i in range(N + 1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.7)
        ax.axvline(x=i, color='lightgray', linewidth=0.5, alpha=0.7)
    


    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect("equal")
    
    # Set integer ticks for all values
    ax.set_xticks(range(N + 1))
    ax.set_yticks(range(N + 1))
    ax.grid(True, which='major', color='gray', linewidth=0.8, alpha=0.5)
    ax.legend()
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Environment Debug (N={N})', fontsize=14)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def _cell_center_xy(r, c, N):
    """
    warehouse uses (r,c) with r=0 at top.
    controller expects (x,y) with y up, origin bottom-left.
    
    Grid lines in warehouse are at integer r, c values.
    Grid line at row r maps to y = N - r (inverted for y-up).
    Cell (r, c) occupies rows [r, r+1) and cols [c, c+1).
    So cell center is at (c + 0.5, N - r - 0.5).
    """
    x = float(c)
    y = float(N - r) # Equivalent to N - r - 0.5
    return (x, y)


def _rect_poly_from_cells(r0, c0, h, w, N):
    """
    Rectangle obstacle in warehouse cell coords -> polygon corners in controller coords.
    Returns points in (x,y), closed not required.
    The rectangle spans from (c0, r0) to (c0+w, r0+h) in grid coordinates.
    """
    x0 = float(c0)
    x1 = float(c0 + w)
    # convert r (top-down) to y (bottom-up)
    # r0 is top of rectangle, r0+h is bottom
    y_top = float(N - r0)
    y_bot = float(N - (r0 + h))
    return [(x0, y_bot), (x1, y_bot), (x1, y_top), (x0, y_top)]





class Environment:
    """
    Drop-in replacement for your old Environment class.

    Returns:
      grid (N,N) int
      agent_start (x,y) float
      goal (x,y) float
      obstacles: list of polygons, each polygon is list[(x,y),...]
    """
    def __init__(self, N=15, num_rooms=5, num_obstacles=10, seed=None,
                 out_dir=None, add_workstations=False, difficulty = "hard"):
        self.N = int(N)
        self.num_rooms = int(num_rooms)
        self.num_obstacles = int(num_obstacles)
        self.seed = seed
        self.out_dir = out_dir
        self.add_workstations = add_workstations
        self.difficulty = difficulty


        # will be filled after generation
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = None

    def create_environment(self):
        # 1) Generate using warehouse.py (Code-2). It saves png+json and returns paths.
        out_png, out_json = generate_env(
            N=self.N,
            num_rooms=self.num_rooms,
            num_obstacles=self.num_obstacles,
            seed=self.seed,
            out_dir=self.out_dir,
            add_workstations=self.add_workstations,
            difficulty=self.difficulty,
        )

        # 2) Load meta
        with open(out_json, "r") as f:
            meta = json.load(f)

        N = int(meta["N"])

        # 3) Agent/goal to (x,y) in [0..N]
        ar, ac = int(meta["agent"]["r"]), int(meta["agent"]["c"])
        gr, gc = int(meta["goal"]["r"]), int(meta["goal"]["c"])
        agent_xy = _cell_center_xy(ar, ac, N)
        goal_xy = _cell_center_xy(gr, gc, N)

        # 4) Obstacles: rectangle obstacles + walls (rooms) + outer boundary
        obstacles_polys = []

        # rectangle obstacles
        for o in meta["obstacles"]:
            obstacles_polys.append(_rect_poly_from_cells(o["r0"], o["c0"], o["h"], o["w"], N))

        # room + boundary walls directly from warehouse grids (NO buffering)
        vwall = meta["vwall_room"]
        hwall = meta["hwall_room"]
        obstacles_polys.extend(walls_from_warehouse(vwall, hwall, N))

        # outer boundary as thin walls
        t = 0.001
        # bottom y=0, top y=N, left x=0, right x=N
        # obstacles_polys.append(_thin_rect_from_segment(0, 0, N, 0, t=t))
        # obstacles_polys.append(_thin_rect_from_segment(0, N, N, N, t=t))
        # obstacles_polys.append(_thin_rect_from_segment(0, 0, 0, N, t=t))
        # obstacles_polys.append(_thin_rect_from_segment(N, 0, N, N, t=t))

        # 5) Grid (N,N): mark obstacle rectangles + any cell touched by a wall polygon
        # Keep it simple: mark obstacle rectangle cells as 1.
        grid = np.zeros((N, N), dtype=int)
        for o in meta["obstacles"]:
            r0, c0, h, w = o["r0"], o["c0"], o["h"], o["w"]
            grid[r0:r0 + h, c0:c0 + w] = 1

        # Convert grid to controller orientation (y up):
        # Controller doesn't index grid directly usually, but lidar might.
        # Make grid[y,x] with y up like Code-1 by flipping rows.
        grid_y_up = np.flipud(grid)

        self.grid = grid_y_up
        self.agent_pos = agent_xy
        self.goal_pos = goal_xy
        self.obstacles = obstacles_polys

        debug_render_environment(
            obstacles_polys,
            agent_xy,
            goal_xy,
            N,
            out_path=(self.out_dir + "/env_debug_env_py.png" if self.out_dir else "env_debug_env_py.png")
        )


        return self.grid, self.agent_pos, self.goal_pos, self.obstacles

    def set_agent_and_goal(self, agent_pos, goal_pos):
        self.agent_pos = tuple(agent_pos)
        self.goal_pos = tuple(goal_pos)

    def set_obstacles(self, obstacle_coords):
        # optional, if you still use it somewhere
        self.obstacles = obstacle_coords
if __name__ == "__main__":
    print("Running environment.py standalone")

    # SAME parameters as controller
    N = 15
    num_rooms = 3
    num_obstacles = 10
    difficulty = "medium"
    different = False
    fixed_seed = 15 if not different else None

    env = Environment(
        N=N,
        num_rooms=num_rooms,
        num_obstacles=num_obstacles,
        seed=fixed_seed,
        out_dir=".",
        difficulty=difficulty,
    )

    grid, agent, goal, obs_polys = env.create_environment()

    print("Agent:", agent)
    print("Goal :", goal)
    print("Num obstacles:", len(obs_polys))