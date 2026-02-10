# controller.py

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from shapely.geometry import Polygon, LineString, Point

from environment import Environment
from lidar_obstacle import LidarObstacleDetector
from logger import Logger
from LLM import get_detour_waypoints


class NavigationController:
    def __init__(self):
        # --- environment ---
        self.env = Environment()
        self.grid, self.agent_start, self.goal, self.obstacles = self.env.create_environment()

        # --- lidar ---
        self.lidar = LidarObstacleDetector(self.grid, self.obstacles)
        self.range_limit = 0.5  # keep in sync with lidar

        # --- motion ---
        self.dt = 0.1
        self.k = 1.0
        self.threshold = 0.1

        # --- geometry: original + buffered (for prompt only) ---
        self.buffer = 0.3  # used for what we SEND to the LLM
        self.polys = [Polygon(pts) for pts in self.obstacles]  # ORIGINAL polys (validation)
        self.buffered_polys = [poly.buffer(self.buffer, join_style=2) for poly in self.polys]  # PROMPT ONLY

        # --- trace & logs ---
        self.positions = [np.array(self.agent_start, dtype=float).copy()]
        self.lidar_data = []
        self.visited_waypoints = []

        self.logger = Logger()
        self.logger.log(f"Environment initialized with agent at {self.agent_start} and goal at {self.goal}")

        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.video_path = os.path.join(self.log_dir, f"navigation_{self.logger.timestamp}.mp4")

    # ======================= MAIN LOOP =======================
    def run(self):
        x = np.array(self.agent_start, dtype=float)
        print(f"Starting navigation from {x.tolist()} to goal at {list(self.goal)}")
        self.logger.log(f"Starting navigation from {x.tolist()} to goal at {list(self.goal)}")

        step_guard = 0
        max_steps = 2000

        while np.linalg.norm(x - self.goal) > self.threshold and step_guard < max_steps:
            step_guard += 1

            # scan first — if anything is hit, stop and query LLM
            hits, obstacle_detected = self._scan_and_detect(x)
            self.positions.append(x.copy())
            self.lidar_data.append((x.copy(), hits))

            if obstacle_detected:
                print(f"Obstacle detected near {x.tolist()} — querying LLM for batch detour waypoints")
                self.logger.log(f"Obstacle detected near {x.tolist()} — querying LLM for batch detour waypoints")

                detour_chain = self._llm_only_detour(x, min_attempts=20)  # up to 20 tries
                if not detour_chain:
                    self.logger.log("LLM produced no usable waypoints after retries — stopping.")
                    print("LLM produced no usable waypoints after retries — stopping.")
                    break

                # follow accepted chain with scan at each micro-step
                for detour_wp in detour_chain:
                    if np.linalg.norm(x - self.goal) <= self.threshold:
                        break
                    self._go_to_waypoint_with_checks(detour_wp)
                    x = self.positions[-1].copy()
                continue

            # if goal is visible, go straight
            if self._goal_visible(x):
                u = -self.k * (x - self.goal)
                u = np.clip(u, -1, 1)
                x = x + u * self.dt
                continue

            # default drift toward goal
            u = -self.k * (x - self.goal)
            u = np.clip(u, -1, 1)
            x = x + u * self.dt

        if np.linalg.norm(x - self.goal) <= self.threshold:
            print("Reached goal!")
            self.logger.log("Reached goal!")
        else:
            print("Stopped (guard or no LLM waypoints).")
            self.logger.log("Stopped (guard or no LLM waypoints).")

        self._create_animation()
        self.logger.close()

    # ======================= LLM-ONLY DETOUR =======================
    def _llm_only_detour(self, x, min_attempts=20):
        """
        Re-queries the LLM until it returns a usable chain, up to min_attempts.
        We pass *buffered obstacles* to the LLM for generation,
        but we validate against the *original obstacles*.
        """
        # build buffered obstacle coords (no closing duplicate point)
        buffered_coords = []
        for poly in self.buffered_polys:
            xs, ys = poly.exterior.xy
            coords = list(zip(xs, ys))
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            buffered_coords.append(coords)

        for attempt in range(1, min_attempts + 1):
            # gentle radius ramp to give model flexibility
            radius = min(0.6 + 0.05 * (attempt - 1), 1.55)

            raw_wps = get_detour_waypoints(
                agent_pos=x.tolist(),
                goal_pos=list(self.goal),
                obstacle_coords=buffered_coords,  # <<<<<< send buffered to LLM
                avoid_list=self.visited_waypoints,
                k_min=5, k_max=8, radius=radius
            )
            self.logger.log(f"[LLM attempt {attempt}] returned {len(raw_wps)} raw wps: {raw_wps}")
            print(f"[LLM attempt {attempt}] {len(raw_wps)} raw detour waypoints")

            valid_chain = self._validate_against_original(x, raw_wps)
            self.logger.log(f"[LLM attempt {attempt}] accepted {len(valid_chain)} wps after validation.")
            print(f"[LLM attempt {attempt}] accepted {len(valid_chain)} waypoints")

            if valid_chain:
                return valid_chain

            # steer the next prompt: avoid everything we just saw
            for wp in raw_wps:
                try:
                    fx = float(wp[0]); fy = float(wp[1])
                    self.visited_waypoints.append([fx, fy])
                except Exception:
                    continue

        return []

    # ======================= VALIDATION (ORIGINAL OBSTACLES) =======================
    def _validate_against_original(self, x, raw_wps):
        """
        Validate waypoints using the ORIGINAL polygons only (no buffer):
        - inside check uses poly.contains(point)
        - segment check rejects if the segment *intersects* the polygon, but allows pure touching.
        """
        valid = []
        prev = x.copy()

        for j, wp in enumerate(raw_wps):
            try:
                wp = np.array([float(wp[0]), float(wp[1])], dtype=float)
            except Exception:
                self.logger.log(f"Reject WP{j}: non-numeric {wp}")
                continue

            # bounds clamp
            wp[0] = np.clip(wp[0], 0.0, 10.0)
            wp[1] = np.clip(wp[1], 0.0, 10.0)

            # reject if point is strictly inside any original obstacle
            if self._point_inside_original(wp):
                self.visited_waypoints.append([float(wp[0]), float(wp[1])])
                self.logger.log(f"Reject WP{j}: inside original obstacle {wp.tolist()}")
                continue

            # reject if the line crosses an obstacle interior (allow touching/grazing)
            if self._segment_crosses_original(prev, wp):
                self.visited_waypoints.append([float(wp[0]), float(wp[1])])
                self.logger.log(
                    f"Reject WP{j}: segment {prev.tolist()} -> {wp.tolist()} intersects original obstacle interior"
                )
                continue

            valid.append(wp)
            prev = wp

        return valid

    def _point_inside_original(self, p):
        pt = Point(float(p[0]), float(p[1]))
        # contains() is false on boundary, which is what we want
        return any(poly.contains(pt) for poly in self.polys)

    def _segment_crosses_original(self, p1, p2):
        seg = LineString([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))])
        # allow pure touching; reject if intersects but not just touching
        for poly in self.polys:
            if seg.intersects(poly) and not seg.touches(poly):
                return True
        return False

    # ======================= OTHER HELPERS =======================
    def _scan_and_detect(self, x):
        hits = self.lidar.scan(x)
        obstacle_detected = False
        for angle, end in hits:
            expected_end = np.array([
                x[0] + self.range_limit * np.cos(angle),
                x[1] + self.range_limit * np.sin(angle)
            ])
            if not np.allclose(end, expected_end, atol=1e-2):
                obstacle_detected = True
                break
        return hits, obstacle_detected

    def _goal_visible(self, p):
        seg = LineString([(float(p[0]), float(p[1])), (float(self.goal[0]), float(self.goal[1]))])
        # visibility is also checked against ORIGINAL obstacles now
        return not any(seg.intersects(poly) and not seg.touches(poly) for poly in self.polys)

    def _go_to_waypoint_with_checks(self, detour_wp):
        x = self.positions[-1].copy()
        guard = 0
        while np.linalg.norm(x - detour_wp) > self.threshold and guard < 600:
            guard += 1

            # if path to current detour crosses original obstacle interior, stop using it
            if self._segment_crosses_original(x, detour_wp):
                self.logger.log(f"Abort moving to {detour_wp.tolist()} — segment crosses original obstacle")
                break

            u = -self.k * (x - detour_wp)
            u = np.clip(u, -1, 1)
            x = x + u * self.dt

            hits, _ = self._scan_and_detect(x)
            self.positions.append(x.copy())
            self.lidar_data.append((x.copy(), hits))

            if self._goal_visible(x):
                self.logger.log("Goal visible during detour step; switching to straight navigation")
                break

    # ======================= ANIMATION =======================
    def _create_animation(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_title("Navigation with LIDAR (LLM-only detours)")

        # draw original obstacles (what we validate against)
        for obs in self.obstacles:
            rect = patches.Polygon(obs, closed=True, color='black')
            ax.add_patch(rect)

        agent_dot, = ax.plot([], [], 'go', markersize=8)
        ax.plot(self.goal[0], self.goal[1], 'b*', markersize=12)
        path_line, = ax.plot([], [], 'k-', linewidth=1.5)

        lidar_lines = []
        collision_markers = []

        def init():
            agent_dot.set_data([], [])
            path_line.set_data([], [])
            return [agent_dot, path_line]

        def update(frame):
            for ln in lidar_lines:
                ln.remove()
            lidar_lines.clear()
            for m in collision_markers:
                m.remove()
            collision_markers.clear()

            pos, hits = self.lidar_data[frame]
            agent_dot.set_data([pos[0]], [pos[1]])
            path_line.set_data([p[0] for p in self.positions[:frame+1]],
                               [p[1] for p in self.positions[:frame+1]])

            for angle, end in hits:
                expected_end = np.array([
                    pos[0] + self.range_limit * np.cos(angle),
                    pos[1] + self.range_limit * np.sin(angle)
                ])
                is_hit = not np.allclose(end, expected_end, atol=1e-2)
                color = 'r' if is_hit else 'y'

                line = ax.plot([pos[0], end[0]], [pos[1], end[1]], color=color, linewidth=1)[0]
                lidar_lines.append(line)

                if is_hit:
                    marker = ax.plot(end[0], end[1], 'rx', markersize=8, mew=1.5)[0]
                    collision_markers.append(marker)

            return [agent_dot, path_line] + lidar_lines + collision_markers

        ani = FuncAnimation(fig, update, frames=len(self.lidar_data),
                            init_func=init, blit=False, interval=80)

        writer = FFMpegWriter(fps=12)
        ani.save(self.video_path, writer=writer)
        plt.close()


if __name__ == "__main__":
    nav = NavigationController()
    nav.run()
