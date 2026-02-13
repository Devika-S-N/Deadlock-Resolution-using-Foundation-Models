# memory.py
import os
import json  # <-- added
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


class Memory:
    """
    Separate, live heatmap + obstacle clustering (does NOT touch main video).

    Blue layer = agent trail heat
    Red  layer = obstacle hit heat
    Red dots   = individual LIDAR hit points
    Lime dot   = agent
    Blue star  = goal

    Usage in controller:
      self.mem = Memory(bounds=(0,10,0,10), cell=0.05, R=0.8,  # 0.8 to avoid merging close obstacles
                        live_viz=True, record=True,
                        log_dir=self.log_dir, run_id=self.logger.timestamp)
      ...
      self.mem.set_goal(goal_x, goal_y)             # once you know the goal
      self.mem.add_scan(x, hits, self.range_limit)  # after each scan
      ...
      self.mem.finalize(logger=self.logger)         # before logger.close()
    """

    def __init__(
        self,
        bounds=(0.0, 10.0, 0.0, 10.0),
        cell=0.05,
        R=0.8,               # assignment radius
        live_viz=True,
        record=True,
        log_dir="logs",
        run_id="run",
        fps=12,
        path_weight=1.0,
        hit_weight=2.0,
        min_cluster_points=3,
        max_points_per_cluster=2000,
        viz_update_every=1,
        rng_seed=17,
        merge_hysteresis=0.6,  # stricter merge than assign (optional)
    ):
        # config
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.cell = float(cell)
        self.R = float(R)
        self.R_merge = float(merge_hysteresis) if merge_hysteresis is not None else self.R
        self.path_w = float(path_weight)
        self.hit_w = float(hit_weight)
        self.min_cluster_points = int(min_cluster_points)
        self.max_pts_per_cluster = int(max_points_per_cluster)
        self.viz_update_every = int(viz_update_every)

        # separate heat layers
        self.w = int(np.ceil((self.xmax - self.xmin) / self.cell))
        self.h = int(np.ceil((self.ymax - self.ymin) / self.cell))
        self.heat_path = np.zeros((self.h, self.w), dtype=np.float32)  # blue
        self.heat_hits = np.zeros((self.h, self.w), dtype=np.float32)  # red

        # clusters
        self.clusters = []
        self._next_id = 0
        self._rng = np.random.default_rng(rng_seed)

        # all hit points for scatter
        self._hit_points = []  # list of [x, y]

        # live viz + recording
        self.live_viz = bool(live_viz)
        self.record = bool(record)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = str(run_id)
        self.heatmap_video_path = os.path.join(self.log_dir, f"heatmap_{self.run_id}.mp4")
        self._vlm_snap_idx = 0


        self._fig = None
        self._ax = None
        self._im_path = None
        self._im_hits = None
        self._hits_scatter = None
        self._agent_scatter = None
        self._goal_scatter = None
        self._coord_text = None
        self._writer = None
        self._setup_done = False
        self._past_detour_points = []     # yellow
        self._current_detour_points = []  # black
        self._step = 0

        if self.live_viz:
            self._setup_viz()
        if self.record:
            self._setup_writer(fps=fps)

    # -------------------- Public API --------------------

    def set_goal(self, gx, gy):
        """Set or update the goal marker on the heatmap."""
        if not self._setup_done:
            return
        if self._goal_scatter is None:
            self._goal_scatter = self._ax.scatter(
                [], [], s=80, marker='*', c='blue',
                edgecolors='k', linewidths=0.7, zorder=5, label="Goal"
            )
        self._goal_scatter.set_offsets(np.array([[float(gx), float(gy)]], dtype=np.float32))
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def add_scan(self, pose, hits, range_limit, atol=1e-2):
        """
        After each LIDAR scan step:
          - update blue path heat
          - extract hit points; update red hit heat + clustering + dots
          - refresh live window
          - record frame (if enabled)
        pose: np.array([x, y])
        hits: list of (angle_rad, end_point np.array([x, y]))
        """
        self._step += 1
        px, py = float(pose[0]), float(pose[1])

        # 1) path (blue)
        self._bump(self.heat_path, px, py, self.path_w)

        # 2) hits (red) — mirror nav logic: a hit if end != expected_end (within atol)
        step_hits = []
        for angle, end in hits:
            ex, ey = float(end[0]), float(end[1])
            step_hits.append((ex, ey))

        for (ex, ey) in step_hits:
            self._hit_points.append([ex, ey])
            self._bump(self.heat_hits, ex, ey, self.hit_w)
            self._add_hit(np.array([ex, ey], dtype=np.float32))

        # 3) draw + record
        if self.live_viz and self._step % self.viz_update_every == 0:
            self._update_viz(px, py)

        if self.record and self._writer is not None and self._fig is not None:
            try:
                self._writer.grab_frame()
            except Exception:
                pass  # never break nav on recording error

    def add_detour_points(self, points):
        """Add VLM detour waypoints to heatmap (yellow dots)."""
        if not points:
            return
        self._detour_points.extend(points)
        if self.live_viz and self._setup_done:
            self._detour_scatter.set_offsets(np.array(self._detour_points, dtype=np.float32))
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()


    def clusters_summary(self):
        """Simple summary for logs."""
        return [
            {
                "id": c["id"],
                "centroid": [float(c["centroid"][0]), float(c["centroid"][1])],
                "bbox": [float(v) for v in c["bbox"]],
                "n_points": int(c["n"]),
            }
            for c in self.clusters
        ]

    def finalize(self, logger=None):
        """Close video writer and optionally log final clusters (with hit points)."""
        if logger is not None:
            clusters = [c for c in self.clusters if c["n"] >= self.min_cluster_points]
            if clusters:
                logger.log("Final obstacle clusters (with LIDAR hit points):")
                for c in clusters:
                    cd = self._cluster_as_dict(c, include_points=True)
                    # One compact JSON-ish line per cluster
                    logger.log(json.dumps(cd))
            else:
                logger.log("Final obstacle clusters: none")
            if self.record:
                logger.log(f"Saved heatmap video to {self.heatmap_video_path}")

        if self._writer is not None:
            try:
                self._writer.finish()
            except Exception:
                pass
            self._writer = None

    # -------------------- Internals: grid --------------------

    def _world_to_grid(self, x, y):
        gx = int((x - self.xmin) / self.cell)
        gy = int((y - self.ymin) / self.cell)
        return gx, gy  # (col, row)

    def _bump(self, grid, x, y, amount):
        gx, gy = self._world_to_grid(x, y)
        if 0 <= gx < self.w and 0 <= gy < self.h:
            grid[gy, gx] += amount  # (row, col) = (y, x)

    # -------------------- Internals: clustering --------------------

    def _add_hit(self, p):
        """Assign hit to nearest cluster by true nearest-point distance; merge with hysteresis."""
        if not self.clusters:
            self._create_cluster(p)
            return

        # nearest-point distance to each cluster (brute-force is fine here)
        dists = []
        for idx, c in enumerate(self.clusters):
            pts = np.vstack(c["points"]) if c["points"] else np.empty((0, 2), dtype=np.float32)
            dmin = float(np.min(np.linalg.norm(pts - p, axis=1))) if pts.size else np.inf
            dists.append((dmin, idx))

        dists.sort(key=lambda t: t[0])
        dmin, best = dists[0]

        if dmin <= self.R:
            merged_into = best
            # merge ONLY clusters really close (hysteresis)
            for k in range(1, len(dists)):
                dk, j = dists[k]
                if dk <= self.R_merge:
                    merged_into = self._merge_clusters(merged_into, j)
                else:
                    break
            self._append_point(self.clusters[merged_into], p)
        else:
            self._create_cluster(p)

    def _create_cluster(self, p):
        color = (1.0, 0.0, 0.0)  # red
        c = {
            "id": self._next_id,
            "centroid": p.copy(),
            "n": 1,
            "bbox": [float(p[0]), float(p[1]), float(p[0]), float(p[1])],
            "points": [p.copy()],
            "color": color,
        }
        self.clusters.append(c)
        self._next_id += 1

    def _merge_clusters(self, i, j):
        if i == j:
            return i
        if j < i:
            i, j = j, i
        ci = self.clusters[i]
        cj = self.clusters[j]

        n_total = ci["n"] + cj["n"]
        new_centroid = (ci["centroid"] * ci["n"] + cj["centroid"] * cj["n"]) / n_total
        x1 = min(ci["bbox"][0], cj["bbox"][0])
        y1 = min(ci["bbox"][1], cj["bbox"][1])
        x2 = max(ci["bbox"][2], cj["bbox"][2])
        y2 = max(ci["bbox"][3], cj["bbox"][3])

        pts = ci["points"] + cj["points"]
        if len(pts) > self.max_pts_per_cluster:
            idxs = np.linspace(0, len(pts) - 1, self.max_pts_per_cluster).astype(int)
            pts = [pts[k] for k in idxs]

        ci["n"] = n_total
        ci["centroid"] = new_centroid
        ci["bbox"] = [x1, y1, x2, y2]
        ci["points"] = pts

        self.clusters.pop(j)
        return i

    def _append_point(self, c, p):
        n = c["n"] + 1
        c["centroid"] = (c["centroid"] * c["n"] + p) / n
        c["n"] = n
        x1, y1, x2, y2 = c["bbox"]
        c["bbox"] = [min(x1, p[0]), min(y1, p[1])], [max(x2, p[0]), max(y2, p[1])]
        # fix bbox assembly to preserve [x1,y1,x2,y2] list
        c["bbox"] = [min(x1, p[0]), min(y1, p[1]), max(x2, p[0]), max(y2, p[1])]
        c["points"].append(p.copy())
        if len(c["points"]) > self.max_pts_per_cluster:
            c["points"] = c["points"][::2]

    # -------------------- Internals: viz --------------------

    def _setup_viz(self):
        if self._setup_done:
            return
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(6, 6))
        self._ax.set_xlim(self.xmin, self.xmax)
        self._ax.set_ylim(self.ymin, self.ymax)
        self._ax.set_aspect('equal', adjustable='box')
        self._ax.set_facecolor('white')
        # make text readable on white bg
        #self._ax.set_title("Live Heatmap (Blue: path, Red: hits)", color='black')

        # use white-based colormaps so empty areas are white in video
        bluemap = LinearSegmentedColormap.from_list("white_blue", [(1, 1, 1), (0, 0, 1)], N=256)
        redmap  = LinearSegmentedColormap.from_list("white_red",  [(1, 1, 1), (1, 0, 0)], N=256)

        # BLUE path layer (under)
        self._im_path = self._ax.imshow(
            self.heat_path,
            origin='lower',
            extent=[self.xmin, self.xmax, self.ymin, self.ymax],
            cmap=bluemap,
            alpha=0.8,
            vmin=0.0, vmax=1.0,
            interpolation='nearest',
            zorder=1,
        )

        # RED hits layer (over)
        self._im_hits = self._ax.imshow(
            self.heat_hits,
            origin='lower',
            extent=[self.xmin, self.xmax, self.ymin, self.ymax],
            cmap=redmap,
            alpha=0.9,
            vmin=0.0, vmax=1.0,
            interpolation='nearest',
            zorder=2,
        )

        # agent (lime) and hit points (red)
        self._agent_scatter = self._ax.scatter([], [], s=50, marker='o', c='lime',
                                               edgecolors='k', linewidths=0.6, zorder=4, label="Agent")
        self._hits_scatter = self._ax.scatter([], [], s=18, marker='.', c='red',
                                              alpha=1.0, zorder=5, label="LIDAR hits")
        # Past detours (yellow)
        self._past_detour_scatter = self._ax.scatter(
            [], [], s=10, marker='o', c='yellow',
            edgecolors='k', linewidths=0.6, zorder=4, label="Past detours"
        )

        # Current detours (black)
        self._current_detour_scatter = self._ax.scatter(
            [], [], s=10, marker='o', c='black',
            edgecolors='k', linewidths=0.6, zorder=5, label="Current detours"
        )

        self._detour_points = []  # storage for all detour WPs


        # goal (blue star) — created lazily in set_goal()
        self._goal_scatter = None

        # coords text (now black for white bg)
        self._coord_text = self._ax.text(
            0.5, 1.02, "", transform=self._ax.transAxes,
            ha="center", va="bottom", fontsize=10, color="black", zorder=6
        )

        # legend (black text on white)
        legend_elems = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=8, linestyle='None', label='Agent'),
            Line2D([0], [0], marker='.', color='red', markersize=10, linestyle='None', label='LIDAR hits'),
            Line2D([0], [0], marker='o', color='black', markersize=8, linestyle='None', label='Detour waypoints'),
            Line2D([0], [0], marker='*', color='blue', markersize=12, linestyle='None', label='Goal'),
        ]
        # leg = self._ax.legend(handles=legend_elems, loc='upper right', fontsize=8,
        #                       facecolor='white', edgecolor='black')
        leg = self._ax.legend(
            handles=legend_elems,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=True
        )
        self._fig.tight_layout()

        for txt in leg.get_texts():
            txt.set_color('black')
        if leg.get_title():
            leg.get_title().set_color('black')

        self._setup_done = True
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _setup_writer(self, fps=12):
        try:
            if self._fig is None:
                return
            self._writer = FFMpegWriter(fps=fps)
            self._writer.setup(self._fig, self.heatmap_video_path, dpi=150)
        except Exception:
            self._writer = None  # recording disabled if ffmpeg missing

    def _update_viz(self, ax_x, ax_y):
        if not self._setup_done:
            return
        try:
            # adaptive contrast per layer
            vmax_path = float(np.percentile(self.heat_path, 98)) if np.any(self.heat_path > 0) else 1.0
            vmax_hits = float(np.percentile(self.heat_hits, 98)) if np.any(self.heat_hits > 0) else 1.0
            vmax_path = max(1.0, vmax_path)
            vmax_hits = max(1.0, vmax_hits)

            self._im_path.set_data(self.heat_path)
            self._im_path.set_clim(vmin=0.0, vmax=vmax_path)

            self._im_hits.set_data(self.heat_hits)
            self._im_hits.set_clim(vmin=0.0, vmax=vmax_hits)

            # agent + hits
            self._agent_scatter.set_offsets(np.array([[ax_x, ax_y]], dtype=np.float32))
            if self._hit_points:
                self._hits_scatter.set_offsets(np.array(self._hit_points, dtype=np.float32))
            else:
                self._hits_scatter.set_offsets(np.empty((0, 2), dtype=np.float32))

            # coords
            self._coord_text.set_text(f"Agent at ({ax_x:.2f}, {ax_y:.2f})")

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            self.live_viz = False  # keep nav running if window closes


    def get_accumulated_lidar_hits(self):
        """
        Get all LIDAR hits accumulated during navigation.
        Returns list of [x, y] coordinates.
        """
        return self._hit_points

    def save_clean_snapshot(self, agent, goal, lidar_hits=None):
        """
        Save a clean snapshot for VLM:
        - White background
        - Red dots: lidar hits
        - Green dot: agent
        - Blue star: goal
        - WITH axes, ticks, grid, and labels
        """

        if lidar_hits is None:
            lidar_hits = self._hit_points

        self._vlm_snap_idx += 1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_aspect('equal')
        ax.set_facecolor('white')

        # ----- axes, ticks, grid -----
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xticks(np.arange(self.xmin, self.xmax + 1, 1.0))
        ax.set_yticks(np.arange(self.ymin, self.ymax + 1, 1.0))
        ax.grid(True, which="both", linewidth=0.5, alpha=0.4)

        # ----- LIDAR hits -----
        if lidar_hits:
            xs, ys = zip(*lidar_hits)
            ax.scatter(xs, ys, c='red', s=25, label="LIDAR hits")

        # ----- agent -----
        ax.scatter(
            agent[0], agent[1],
            c='green', s=100, edgecolors='k', zorder=5, label="Agent"
        )

        # ----- goal -----
        ax.scatter(
            goal[0], goal[1],
            c='blue', marker='*', s=160, edgecolors='k', zorder=6, label="Goal"
        )

        # Past detours (yellow)
        if self._past_detour_points:
            arr = np.array(self._past_detour_points)
            ax.scatter(arr[:, 0], arr[:, 1], c='yellow', s=30, marker='o', label='Past detours')

        # Current detours (black)
        if self._current_detour_points:
            arr = np.array(self._current_detour_points)
            ax.scatter(arr[:, 0], arr[:, 1], c='black', s=30, marker='o', label='Current detours')


        # ax.legend(loc="upper right")
        ax.legend(
            loc='upper left',           # anchor position (top-right of grid area)
            bbox_to_anchor=(1.02, 1),   # move legend just outside the plot
            borderaxespad=0,
            frameon=True
        )
        plt.tight_layout()


        path = os.path.join(
            self.log_dir,
            f"vlm_snapshot_{self.run_id}_{self._vlm_snap_idx:03d}.png"
        )

        plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        return path



    # -------------------- Internals: logging helper --------------------

    def _cluster_as_dict(self, c, include_points=False):
        d = {
            "id": int(c["id"]),
            "n_points": int(c["n"]),
            "centroid": [float(c["centroid"][0]), float(c["centroid"][1])],
            "bbox": [float(v) for v in c["bbox"]],
        }
        if include_points:
            d["points"] = [[float(p[0]), float(p[1])] for p in c["points"]]
        return d

    def archive_current_detours(self):
        """Move current detours to past (black → yellow)."""
        if self._current_detour_points:
            self._past_detour_points.extend(self._current_detour_points)
            self._current_detour_points = []

        if self.live_viz:
            self._past_detour_scatter.set_offsets(
                np.array(self._past_detour_points) if self._past_detour_points else np.empty((0, 2))
            )
            self._current_detour_scatter.set_offsets(np.empty((0, 2)))
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()


    def add_current_detours(self, points):
        """Add a new detour set (black)."""
        self._current_detour_points = list(points)

        if self.live_viz:
            self._current_detour_scatter.set_offsets(
                np.array(self._current_detour_points)
            )
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
