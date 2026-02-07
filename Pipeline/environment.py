# environment.py
import json
import numpy as np
from warehouse import generate_env  # this is your Code-2 file: warehouse.py


def _cell_center_xy(r, c, N):
    """
    warehouse uses (r,c) with r=0 at top.
    controller expects (x,y) with y up, origin bottom-left.
    """
    x = c + 0.5
    y = (N - 1 - r) + 0.5
    return (x, y)


def _rect_poly_from_cells(r0, c0, h, w, N):
    """
    Rectangle obstacle in warehouse cell coords -> polygon corners in controller coords.
    Returns points in (x,y), closed not required.
    """
    x0 = c0
    x1 = c0 + w
    # convert r (top-down) to y (bottom-up)
    y_top = N - r0
    y_bot = N - (r0 + h)
    return [(x0, y_bot), (x1, y_bot), (x1, y_top), (x0, y_top)]


def _thin_rect_from_segment(x0, y0, x1, y1, t=0.06):
    """
    Build a thin rectangle polygon around an axis-aligned wall segment.
    Assumes either horizontal or vertical segment.
    """
    if abs(y1 - y0) < 1e-9:  # horizontal
        y = y0
        xa, xb = sorted([x0, x1])
        return [(xa, y - t/2), (xb, y - t/2), (xb, y + t/2), (xa, y + t/2)]
    else:  # vertical
        x = x0
        ya, yb = sorted([y0, y1])
        return [(x - t/2, ya), (x + t/2, ya), (x + t/2, yb), (x - t/2, yb)]


def _room_walls_as_polys(room, doors, N, thickness=0.06):
    """
    room: dict with r0,c0,h,w
    doors: list of dicts {kind, a, b} (same as meta['doors'])
    Build wall polygons for:
      - top, bottom, left, right
    excluding door segments.
    """
    r0, c0, h, w = room["r0"], room["c0"], room["h"], room["w"]

    # wall lines in grid coordinates
    top_row = r0
    bot_row = r0 + h
    left_col = c0
    right_col = c0 + w

    # Convert to world coordinates:
    # horizontal wall at 'row' is y = N - row, spanning x in [c, c+1]
    # vertical wall at 'col' is x = col, spanning y in [N-(r+1), N-r]
    def y_of_row(row): return N - row
    def x_of_col(col): return col

    # Build door lookup for easy skipping
    door_h = set()  # (row, c)
    door_v = set()  # (col, r)
    for d in doors:
        if d["kind"] == "h":
            door_h.add((d["a"], d["b"]))  # (row, c)
        else:
            door_v.add((d["a"], d["b"]))  # (col, r)

    polys = []

    # --- top wall segments (row = top_row, c from left_col .. right_col-1) ---
    row = top_row
    y = y_of_row(row)
    for c in range(left_col, right_col):
        if (row, c) in door_h:
            continue
        polys.append(_thin_rect_from_segment(c, y, c + 1, y, t=thickness))

    # --- bottom wall segments (row = bot_row) ---
    row = bot_row
    y = y_of_row(row)
    for c in range(left_col, right_col):
        if (row, c) in door_h:
            continue
        polys.append(_thin_rect_from_segment(c, y, c + 1, y, t=thickness))

    # --- left wall segments (col = left_col, r from top_row .. bot_row-1) ---
    col = left_col
    x = x_of_col(col)
    for r in range(top_row, bot_row):
        if (col, r) in door_v:
            continue
        y0 = N - (r + 1)
        y1 = N - r
        polys.append(_thin_rect_from_segment(x, y0, x, y1, t=thickness))

    # --- right wall segments (col = right_col) ---
    col = right_col
    x = x_of_col(col)
    for r in range(top_row, bot_row):
        if (col, r) in door_v:
            continue
        y0 = N - (r + 1)
        y1 = N - r
        polys.append(_thin_rect_from_segment(x, y0, x, y1, t=thickness))

    return polys


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

        # room walls (thin rectangles), respecting doors
        doors = meta.get("doors", [])
        for room in meta.get("rooms", []):
            obstacles_polys.extend(_room_walls_as_polys(room, doors, N, thickness=0.06))

        # outer boundary as thin walls
        t = 0.06
        # bottom y=0, top y=N, left x=0, right x=N
        obstacles_polys.append(_thin_rect_from_segment(0, 0, N, 0, t=t))
        obstacles_polys.append(_thin_rect_from_segment(0, N, N, N, t=t))
        obstacles_polys.append(_thin_rect_from_segment(0, 0, 0, N, t=t))
        obstacles_polys.append(_thin_rect_from_segment(N, 0, N, N, t=t))

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

        return self.grid, self.agent_pos, self.goal_pos, self.obstacles

    def set_agent_and_goal(self, agent_pos, goal_pos):
        self.agent_pos = tuple(agent_pos)
        self.goal_pos = tuple(goal_pos)

    def set_obstacles(self, obstacle_coords):
        # optional, if you still use it somewhere
        self.obstacles = obstacle_coords
