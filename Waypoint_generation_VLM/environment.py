# environment.py

import os
import math
import json
import random
from collections import defaultdict
from typing import List, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
import pandas as pd  # <-- added for Excel logging

from logger import Logger  # shared logger


class Environment:
    """
    Random 10x10 world with polygonal (axis-aligned) obstacles, an agent, and a goal.

    - Samples rectangles, rasterizes to a 10x10 grid, then MERGEs touching/overlapping
      rectangles into rectilinear polygons by tracing the boundary of each connected
      component (4-connected). Redundant collinear points are removed.
    - Obstacles are stored as ordered vertex lists (float world coords).
    - save_environment(tag=...) writes numbered PNG and logs vertices.
    """

    def __init__(
        self,
        grid_size: int = 10,
        seed: Optional[int] = None,
        n_obstacles_range: Tuple[int, int] = (3, 6),
        size_range: Tuple[float, float] = (0.5, 1.5),   # smaller obstacles
        max_fill_ratio: float = 0.35,
        min_agent_goal_dist: float = 4.0,
        retries: int = 200,
        logger: Optional[Logger] = None,                # allow shared logger injection
        log_dir: str = "logs",                          # <-- NEW: configurable output dir
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Use provided log_dir instead of hard-coded "logs"
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # If no shared logger was injected, create one that writes into log_dir
        self.logger = logger if logger is not None else Logger(self.log_dir)

        self.n_obstacles_range = n_obstacles_range
        self.size_range = size_range
        self.max_fill_ratio = max_fill_ratio
        self.min_agent_goal_dist = min_agent_goal_dist
        self.retries = retries

        # Generated each run:
        self.obstacles: List[List[Tuple[float, float]]] = []
        self.agent_pos: Optional[Tuple[float, float]] = None
        self.goal_pos: Optional[Tuple[float, float]] = None

        self._randomize_environment()
        self._log_environment_summary()
        self.logger.log("Environment initialized with randomized, merged-layout obstacles.")

    # ---------------- Public API ----------------

    def create_environment(self):
        """Return occupancy grid, agent start, goal, and obstacles (merged polygons)."""
        self.logger.log("Environment (grid, agent, goal, merged obstacles) returned.")
        return self.grid, self.agent_pos, self.goal_pos, self.obstacles

    def set_agent_and_goal(self, agent_pos, goal_pos):
        self.agent_pos = tuple(agent_pos)
        self.goal_pos = tuple(goal_pos)
        self.logger.log(
            f"Agent position set to {self.agent_pos}, Goal position set to {self.goal_pos}"
        )

    def set_obstacles(self, obstacle_coords):
        """
        Optional: set custom obstacles (grid cells) and update grid.
        obstacle_coords: iterable of (x, y) integer grid cells to mark occupied.
        """
        for (x, y) in obstacle_coords:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[y, x] = 1
        # Re-extract merged polygons from grid
        self.obstacles = self._grid_to_polygons(self.grid)
        self.logger.log("Custom obstacles set (cells). Recomputed merged polygons.")

    def save_environment(self, tag: Optional[Union[int, str]] = None):
        """
        Save two PNGs:
        - Unlabelled copy in parent_images_log/
        - Labelled copy in parent_images_log/labelled/
        """
        # Resolve tag for filenames
        if tag is None:
            resolved_tag = str(getattr(self.logger, "timestamp", "run"))
        else:
            if isinstance(tag, int) or (isinstance(tag, str) and tag.isdigit()):
                resolved_tag = f"{int(tag):02d}"
            else:
                resolved_tag = str(tag).strip()

        # Base fig
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")

        # Draw obstacles
        for obs in self.obstacles:
            patch = MplPolygon(
                obs, closed=True, facecolor="black", edgecolor="black", linewidth=1.0
            )
            ax.add_patch(patch)

        # Plot agent & goal
        if self.agent_pos:
            ax.plot(self.agent_pos[0], self.agent_pos[1], "o", color="blue", markersize=8,
                    markeredgecolor="k", markeredgewidth=0.6, label="Agent")
        if self.goal_pos:
            ax.plot(self.goal_pos[0], self.goal_pos[1], "*", color="green", markersize=12,
                    markeredgecolor="k", markeredgewidth=0.6, label="Goal")

        # Grid
        ticks = np.arange(0, self.grid_size + 1, 1.0)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.grid(True, which="major", linewidth=1.2, color="#d9d9d9")
        ax.set_xticks(np.arange(0, self.grid_size + 0.5, 0.5), minor=True)
        ax.set_yticks(np.arange(0, self.grid_size + 0.5, 0.5), minor=True)
        #ax.grid(True, which="minor", linewidth=0.6, color="#efefef", alpha=0.9)

        leg = ax.legend(loc="upper right", fontsize=6, facecolor="white", edgecolor="black")
        for txt in leg.get_texts():
            txt.set_color("black")

        plt.tight_layout()

        # ----------- Save unlabelled -----------
        unlabelled_path = os.path.join(self.log_dir, f"environment_{resolved_tag}.png")
        plt.savefig(unlabelled_path, dpi=200, bbox_inches="tight")

        # ----------- Add labels and save labelled -----------
        # Agent + Goal
        if self.agent_pos:
            ax.annotate(f"A: ({self.agent_pos[0]:.2f},{self.agent_pos[1]:.2f})",
                        xy=self.agent_pos, xytext=(6, 8), textcoords="offset points",
                        fontsize=6, color="blue",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        if self.goal_pos:
            ax.annotate(f"G: ({self.goal_pos[0]:.2f},{self.goal_pos[1]:.2f})",
                        xy=self.goal_pos, xytext=(6, -12), textcoords="offset points",
                        fontsize=6, color="green",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        # Obstacle vertices
        for oi, obs in enumerate(self.obstacles, start=1):
            for vi, (x, y) in enumerate(obs, start=1):
                ax.annotate(f"O{oi}v{vi}\n({x:.1f},{y:.1f})",
                            xy=(x, y), xytext=(4, 4), textcoords="offset points",
                            fontsize=5, color="red",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

        labelled_dir = os.path.join(self.log_dir, "labelled")
        os.makedirs(labelled_dir, exist_ok=True)
        labelled_path = os.path.join(labelled_dir, f"environment_{resolved_tag}_labelled.png")
        plt.savefig(labelled_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        self.logger.log(f"[ENV {resolved_tag}] unlabelled: {unlabelled_path}")
        self.logger.log(f"[ENV {resolved_tag}] labelled:   {labelled_path}")
        self._log_obstacle_vertices(tag=resolved_tag)

        return unlabelled_path, labelled_path


    # ---------------- Randomization internals ----------------

    def _randomize_environment(self):
        """Create randomized rectangles, rasterize, MERGE, then place agent/goal."""
        # 1) sample rectangles (float polygons)
        raw_rects = self._sample_obstacles()
        # 2) rasterize rectangles to coarse grid
        self.grid = self._polygons_to_grid(raw_rects)
        # 3) extract merged polygons from the raster (connected components → boundary loops)
        self.obstacles = self._grid_to_polygons(self.grid)
        # 4) place agent/goal in free space (respect merged polygons)
        self.agent_pos, self.goal_pos = self._sample_agent_and_goal(self.obstacles)

    def _sample_obstacles(self) -> List[List[Tuple[float, float]]]:
        """
        Sample a random number of axis-aligned rectangular polygons.
        We allow overlap; later union happens in _grid_to_polygons.
        """
        n_min, n_max = self.n_obstacles_range
        n_target = random.randint(n_min, n_max)

        polys: List[List[Tuple[float, float]]] = []
        attempt = 0

        def estimated_fill(polys_) -> float:
            grid = self._polygons_to_grid(polys_)
            return grid.mean()

        while len(polys) < n_target and attempt < self.retries:
            attempt += 1
            w = random.uniform(*self.size_range)
            h = random.uniform(*self.size_range)

            x0 = random.uniform(0.0, self.grid_size - w)
            y0 = random.uniform(0.0, self.grid_size - h)

            rect = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]

            # Keep global fill under cap to avoid over-clutter
            candidate = polys + [rect]
            if estimated_fill(candidate) <= self.max_fill_ratio:
                polys.append(rect)

        return polys

    def _sample_agent_and_goal(self, obstacles_polys):
        """
        Sample agent & goal at *integer* interior intersections (no boundaries),
        not inside obstacles and not on obstacle edges, with min separation.
        """
        paths = [MplPath(np.array(p)) for p in obstacles_polys]

        # Count boundary as "inside" by slightly dilating polygons (radius > 0)
        def is_free_int(pt):
            x, y = pt
            # must be interior (no environment boundary points)
            if not (1 <= x <= self.grid_size - 1 and 1 <= y <= self.grid_size - 1):
                return False
            # forbid inside or on edges (radius catches boundary)
            return not any(path.contains_point((x, y), radius=1e-9) for path in paths)

        a = g = None
        for _ in range(self.retries):
            a_candidate = (
                random.randint(1, self.grid_size - 1),
                random.randint(1, self.grid_size - 1),
            )
            if not is_free_int(a_candidate):
                continue

            for _ in range(self.retries // 2):
                g_candidate = (
                    random.randint(1, self.grid_size - 1),
                    random.randint(1, self.grid_size - 1),
                )
                if is_free_int(g_candidate) and self._euclid(a_candidate, g_candidate) >= self.min_agent_goal_dist:
                    a, g = a_candidate, g_candidate
                    break
            if a and g:
                break

        if a is None or g is None:
            self.logger.log("Could not place integer agent/goal safely; falling back.")
            a = (1, 1)
            g = (self.grid_size - 1, self.grid_size - 1)

        return a, g


    # ---------------- Rasterization & Polygonization ----------------

    def _polygons_to_grid(self, polygons: List[List[Tuple[float, float]]]) -> np.ndarray:
        """
        Mark a 10x10 grid cell as occupied if its center is inside any polygon.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        if not polygons:
            return grid

        paths = [MplPath(np.array(poly)) for poly in polygons]
        xs = np.arange(0.5, self.grid_size, 1.0)
        ys = np.arange(0.5, self.grid_size, 1.0)
        xv, yv = np.meshgrid(xs, ys)
        pts = np.vstack([xv.ravel(), yv.ravel()]).T

        inside = np.zeros(pts.shape[0], dtype=bool)
        for path in paths:
            inside |= path.contains_points(pts)

        grid[yv.astype(int).ravel(), xv.astype(int).ravel()] = inside.astype(int)
        return grid

    def _grid_to_polygons(self, grid: np.ndarray) -> List[List[Tuple[float, float]]]:
        """
        Convert an occupancy grid to one or more rectilinear polygons by tracing
        boundary edges of each connected component (4-connected). Vertices are
        integer-aligned (cell boundaries), ordered, and collinear points removed.
        """
        H, W = grid.shape
        # 1) collect boundary segments on grid lines around occupied cells
        edges = set()  # each edge is ((x1,y1),(x2,y2)) with integer coords
        for y in range(H):
            for x in range(W):
                if grid[y, x] == 1:
                    # bottom
                    if y - 1 < 0 or grid[y - 1, x] == 0:
                        edges.add(((x, y), (x + 1, y)))
                    # top
                    if y + 1 >= H or grid[y + 1, x] == 0:
                        edges.add(((x, y + 1), (x + 1, y + 1)))
                    # left
                    if x - 1 < 0 or grid[y, x - 1] == 0:
                        edges.add(((x, y), (x, y + 1)))
                    # right
                    if x + 1 >= W or grid[y, x + 1] == 0:
                        edges.add(((x + 1, y), (x + 1, y + 1)))

        # 2) build adjacency (undirected) among boundary vertices
        from collections import defaultdict
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

        # 3) trace loops while edges remain
        def pop_loop(start_point):
            loop = [start_point]
            prev = None
            cur = start_point
            while True:
                nbrs = [n for n in adj[cur] if (cur, n) in edges or (n, cur) in edges]
                if not nbrs:
                    break  # safety
                nxt = nbrs[0] if prev is None or len(nbrs) == 1 else (nbrs[0] if nbrs[1] == prev else nbrs[1])
                edges.discard((cur, nxt)); edges.discard((nxt, cur))
                if nxt in adj[cur]: adj[cur].remove(nxt)
                if cur in adj[nxt]: adj[nxt].remove(cur)
                if nxt == start_point:
                    break
                loop.append(nxt)
                prev, cur = cur, nxt
            return loop

        polygons: List[List[Tuple[float, float]]] = []
        while edges:
            endpoints = [p for e in list(edges) for p in e]
            start = min(endpoints, key=lambda p: (p[1], p[0]))
            raw_loop = pop_loop(start)
            if raw_loop[0] != raw_loop[-1]:
                raw_loop.append(raw_loop[0])

            # simplify collinear points
            def collinear(p0, p1, p2):
                x0, y0 = p0; x1, y1 = p1; x2, y2 = p2
                return (x1 - x0) * (y2 - y1) == (y1 - y0) * (x2 - x1)

            simplified = []
            for pt in raw_loop:
                simplified.append(pt)
                while len(simplified) >= 3 and collinear(simplified[-3], simplified[-2], simplified[-1]):
                    simplified.pop(-2)

            if len(simplified) > 1 and simplified[0] == simplified[-1]:
                simplified.pop()

            poly = [(float(x), float(y)) for (x, y) in simplified]
            polygons.append(poly)

        # ensure clockwise orientation (optional)
        def signed_area(poly):
            s = 0.0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                s += x1 * y2 - x2 * y1
            return 0.5 * s

        oriented = []
        for poly in polygons:
            if signed_area(poly) < 0:
                oriented.append(poly)           # already clockwise
            else:
                oriented.append(list(reversed(poly)))
        return oriented

    # ---------------- Logging helpers ----------------

    def _log_environment_summary(self):
        """Write a concise, parseable summary to the rolling log."""
        self.logger.log(
            f"Agent: ({self.agent_pos[0]:.2f}, {self.agent_pos[1]:.2f}); "
            f"Goal: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f}); "
            f"Obstacles (merged): {len(self.obstacles)}"
        )
        self._log_obstacle_vertices()

    def _log_obstacle_vertices(self, tag: Optional[str] = None):
        """Write obstacle vertices to the log for traceability (merged polygons)."""
        prefix = f"[ENV {tag}] " if tag else ""
        if not self.obstacles:
            self.logger.log(prefix + "Obstacle vertices (merged): []")
            return
        lines = []
        for i, poly in enumerate(self.obstacles, start=1):
            verts = ", ".join(f"({x:.2f},{y:.2f})" for (x, y) in poly)
            lines.append(f"{prefix}O{i}: [{verts}]")
        self.logger.log("\n".join(lines))

    # ---------------- Geometry helpers ----------------

    @staticmethod
    def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------- Batch helpers for images + Excel ----------------

def _fmt_point(pt: Tuple[float, float]) -> str:
    return f"({pt[0]:.2f}, {pt[1]:.2f})"

def _fmt_poly(poly: List[Tuple[float, float]]) -> str:
    return "[" + ", ".join(_fmt_point(p) for p in poly) + "]"

def generate_random_images(
    num: int = 10,
    out_dir: str = "parent_images_log",
    seed: Optional[int] = None
) -> str:
    """
    Generate `num` randomized environments as PNGs into `out_dir`,
    and write an Excel sheet 'environments_log.xlsx' with columns:
      agent position, goal position, obstacle1, obstacle2, ...
    Returns the Excel path.
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    # (optional) make results reproducible if seed provided
    base_seed = seed if seed is not None else random.randint(0, 10_000_000)

    for i in range(1, num + 1):
        print(f"Generating image {i}")
        # vary the seed per image so each environment differs
        env = Environment(seed=base_seed + i, log_dir=out_dir)
        # tag images 01, 02, ...
        env.save_environment(tag=i)

        # build one row for the Excel
        row = {
            "agent position": _fmt_point(env.agent_pos),
            "goal position": _fmt_point(env.goal_pos),
        }
        for j, poly in enumerate(env.obstacles, start=1):
            row[f"obstacle{j}"] = _fmt_poly(poly)

        rows.append(row)

    # Normalize columns so all rows share the same set (pads missing obstacleN with NaN)
    df = pd.DataFrame(rows)

    excel_path = os.path.join(out_dir, "environments_log.xlsx")
    df.to_excel(excel_path, index=False)
    return excel_path


# ---------------- Script entrypoint ----------------

if __name__ == "__main__":
    # Produce 10 images + Excel into ./parent_images_log
    excel_path = generate_random_images(num=100, out_dir="new_minor_grid_off")
    print(f"Wrote Excel log to: {excel_path}")
