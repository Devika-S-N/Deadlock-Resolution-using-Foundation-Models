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
from detour_planner import get_detour_waypoints
from memory import Memory


class NavigationController:
    def __init__(self):
        # --- environment ---
        self.env = Environment()
        self.grid, self.agent_start, self.goal, self.obstacles = self.env.create_environment()
        # grid is (N,N). Use that N for all plots
        self.N = int(self.grid.shape[0])
        self.in_escape_mode = False
        self.escape_timer = 0
        self.following_wall = False
        self.wall_follow_dir = None


        # --- logger (create once, share everywhere) ---
        self.logger = Logger(log_dir="logs", strict_order=True)
        self.logger.log(f"Environment initialized with agent at {self.agent_start} and goal at {self.goal}")

        # --- lidar (use the SAME logger) ---
        self.lidar = LidarObstacleDetector(
            self.grid, self.obstacles,
            resolution=0.1, range_limit=0.5, angle_increment=np.deg2rad(10),
            logger=self.logger
        )
        self.range_limit = 0.5  # keep in sync with lidar

        # --- motion ---
        self.dt = 0.1
        self.k = 1.0
        self.threshold = 0.1

        # --- geometry: original + buffered (for prompt only) ---
        self.buffer = 0.3  # used for what we SEND to the  
        self.polys = [Polygon(pts) for pts in self.obstacles]  # ORIGINAL polys (validation)
        self.buffered_polys = [poly.buffer(self.buffer, join_style=2) for poly in self.polys]  # PROMPT ONLY

        # --- trace & logs ---
        self.positions = [np.array(self.agent_start, dtype=float).copy()]
        self.lidar_data = []
        self.visited_waypoints = []

        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.video_path = os.path.join(self.log_dir, f"navigation_{self.logger.timestamp}.mp4")

        # --- memory initialisation ---
        self.mem = Memory(bounds=(0, self.N, 0, self.N), cell=0.05, R=0.8, live_viz=True, record=True, log_dir=self.log_dir, run_id=self.logger.timestamp)
        self.mem.set_goal(self.goal[0], self.goal[1])  # <<< add this line


    # ======================= MAIN LOOP =======================
    def run(self):
        x = np.array(self.agent_start, dtype=float)
        print(f"Starting navigation from {x.tolist()} to goal at {list(self.goal)}")
        self.logger.log(f"Starting navigation from {x.tolist()} to goal at {list(self.goal)}")

        step_guard = 0
        max_steps = 2000

        while np.linalg.norm(x - self.goal) > self.threshold and step_guard < max_steps:
            step_guard += 1

            # --- scan first: bracket with begin/end so ordering is obvious ---
            self.logger.log(f"Scan step begin at {x.tolist()}")
            hits, obstacle_detected, red_hits = self._scan_and_detect(x)

            # ---------------- WALL FOLLOWING LOGIC ----------------
            if red_hits:
                # we have LIDAR hits → enter wall-follow mode
                if not self.following_wall:
                    print(" Wall detected — switching to wall-follow mode.")
                    self.following_wall = True

                pts = np.array(red_hits)
                # compute tangent direction (principal component)
                if len(pts) >= 2:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca.fit(pts - np.mean(pts, axis=0))
                    tangent = pca.components_[0] / np.linalg.norm(pca.components_[0])
                    self.wall_follow_dir = tangent
                else:
                    # fallback direction if only one hit
                    self.wall_follow_dir = np.array([1.0, 0.0])

            else:
                # no LIDAR hits → door/open space reached
                if self.following_wall:
                    print("🚪 Doorway detected — resuming goal attraction.")
                self.following_wall = False
# ------------------------------------------------------

            # ---------------- ESCAPE MODE DETECTION ----------------
            num_hits = len(red_hits)
            if num_hits > 0:
                hit_angles = np.arctan2(
                    [h[1] - x[1] for h in red_hits],
                    [h[0] - x[0] for h in red_hits]
                )
                angular_span = np.degrees(np.ptp(hit_angles))
            else:
                angular_span = 0

            # enter escape mode if enclosed by walls
            if (num_hits > 10 and angular_span > 120):
                if not self.in_escape_mode:
                    print(" Entering ESCAPE mode — ignoring global goal temporarily.")
                self.in_escape_mode = True
                self.escape_timer = 30  # stay in escape mode for next 30 cycles
            else:
                if self.in_escape_mode:
                    self.escape_timer -= 1
                    if self.escape_timer <= 0 and angular_span < 100:
                        print(" Environment open — resuming goal attraction.")
                        self.in_escape_mode = False
            # --------------------------------------------------------



            self.logger.log(f"Scan step end: obstacle_detected={obstacle_detected}")

            self.positions.append(x.copy())
            self.lidar_data.append((x.copy(), hits))

            self.mem.add_scan(x, hits, self.range_limit)

            if obstacle_detected:
                # pick representative hit (closest to robot)
                if len(red_hits) > 0:
                    dists = [np.linalg.norm(np.array(hit) - x) for hit in red_hits]
                    nearest_hit = red_hits[np.argmin(dists)]
                    obstacle_info = f"Obstacle detected near {np.round(nearest_hit, 2).tolist()} (from LIDAR hits)"
                else:
                    obstacle_info = "Obstacle detected (no red hits recorded)"

                print(obstacle_info + " — querying for batch detour waypoints")
                self.logger.log(obstacle_info + " — querying for batch detour waypoints")

                detour_chain = self._llm_only_detour(x, min_attempts=20)

                if not detour_chain:
                    self.logger.log("  produced no usable waypoints after retries — stopping.")
                    print("  produced no usable waypoints after retries — stopping.")
                    break

                # follow accepted chain with scan at each micro-step
                for detour_wp in detour_chain:
                    if np.linalg.norm(x - self.goal) <= self.threshold:
                        break
                    self._go_to_waypoint_with_checks(detour_wp)
                    x = self.positions[-1].copy()
                continue

            # if in escape mode, follow detour waypoints only
            if self.in_escape_mode and 'detour_chain' in locals() and detour_chain:
                target = detour_chain[-1]
            else:
                target = self.goal

            # move toward target if visible
            if self.following_wall and self.wall_follow_dir is not None:
                # move parallel to wall tangent
                move_dir = self.wall_follow_dir
                u = move_dir / np.linalg.norm(move_dir)
                print(f" Following wall tangent: {np.round(move_dir, 2)}")
            else:
                # normal goal attraction
                u = -self.k * (x - self.goal)
                print(" Moving toward global goal")

            if self.following_wall:
                u = 0.7 * self.wall_follow_dir + 0.3 * (-self.k * (x - self.goal))


            # apply velocity
            u = np.clip(u, -1, 1)
            x = x + u * self.dt


        if np.linalg.norm(x - self.goal) <= self.threshold:
            print("Reached goal!")
            self.logger.log("Reached goal!")
        else:
            print("Stopped (guard or no   waypoints).")
            self.logger.log("Stopped (guard or no   waypoints).")

        self._create_animation()
        self.mem.finalize(logger=self.logger) 
        self.logger.close()


    # ======================= LLM DETOUR REPLACEMENT WITH   =======================
    def _llm_only_detour(self, x, min_attempts=1):
        """
        Local   detour planner using only LIDAR hit points.
        """
        # Perform a fresh scan at current position
        hits, _, red_hits = self._scan_and_detect(x)

        

        # Run   on local grid using LIDAR hits
        # --- Run   using the new image-based detour planner ---
        # detour_points = get_detour_waypoints(
        #     start=x,
        #     goal=self.goal,
        #     lidar_hits=hits,
        #     offset=0.25,
        #     step=0.15
        # )
        detour_points = get_detour_waypoints(start=x, goal=self.goal, lidar_hits=[(None, p) for p in red_hits])



        if not detour_points:
            self.logger.log("Local   failed to find detour.")
            print("Local   failed to find detour.")
            return []

        valid_chain = [np.array(wp, dtype=float) for wp in detour_points]
        self.logger.log(f"Local   detour generated {len(valid_chain)} waypoints.")
        print(f"Local   detour generated {len(valid_chain)} waypoints.")
        return valid_chain



    # ======================= VALIDATION (ORIGINAL OBSTACLES) =======================
    def _validate_against_original(self, x, raw_wps):
        valid = []
        prev = x.copy()

        for j, wp in enumerate(raw_wps, start=1):
            try:
                wp = np.array([float(wp[0]), float(wp[1])], dtype=float)
            except Exception:
                self.logger.log(f"Reject WP{j}: non-numeric {wp}")
                continue

            wp[0] = np.clip(wp[0], 0.0, float(self.N))
            wp[1] = np.clip(wp[1], 0.0, float(self.N))


            if self._point_inside_original(wp):
                self.visited_waypoints.append([float(wp[0]), float(wp[1])])
                self.logger.log(f"Reject WP{j}: inside original obstacle {wp.tolist()}")
                continue

            if self._segment_crosses_original(prev, wp):
                self.visited_waypoints.append([float(wp[0]), float(wp[1])])
                self.logger.log(
                    f"Reject WP{j}: segment {prev.tolist()} -> {wp.tolist()} intersects original obstacle interior"
                )
                continue

            self.logger.log(f"Accept WP{j}: {wp.tolist()}")
            valid.append(wp)
            prev = wp

        return valid

    def _point_inside_original(self, p):
        pt = Point(float(p[0]), float(p[1]))
        return any(poly.contains(pt) for poly in self.polys)

    def _segment_crosses_original(self, p1, p2):
        seg = LineString([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))])
        for poly in self.polys:
            if seg.intersects(poly) and not seg.touches(poly):
                return True
        return False

    # ======================= OTHER HELPERS =======================
    def _scan_and_detect(self, x):
        hits = self.lidar.scan(x)
        obstacle_detected = False
        red_hits = []  # store true hit points (used for detour planner)

        for angle, end in hits:
            expected_end = np.array([
                x[0] + self.range_limit * np.cos(angle),
                x[1] + self.range_limit * np.sin(angle)
            ])
            is_hit = not np.allclose(end, expected_end, atol=1e-2)
            if is_hit:
                red_hits.append(end)
                obstacle_detected = True

        # optional: log for debugging
        if self.logger:
            self.logger.log(f"LIDAR red hits (obstacles): {np.round(red_hits, 3).tolist()}")

        return hits, obstacle_detected, red_hits


    def _goal_visible(self, p):
        seg = LineString([(float(p[0]), float(p[1])), (float(self.goal[0]), float(self.goal[1]))])
        return not any(seg.intersects(poly) and not seg.touches(poly) for poly in self.polys)

    def _go_to_waypoint_with_checks(self, detour_wp):
        # (keep your existing movement logging or the preflight version we discussed earlier)
        self.logger.log(f"Moving to waypoint {detour_wp.tolist() if hasattr(detour_wp, 'tolist') else list(detour_wp)}")

        x = self.positions[-1].copy()
        guard = 0
        crossed_blocked = False
        lidar_blocked = False
        no_progress_count = 0
        last_dist = np.linalg.norm(x - detour_wp)
        max_no_progress = 15

        while np.linalg.norm(x - detour_wp) > self.threshold and guard < 600:
            guard += 1

            if self._segment_crosses_original(x, detour_wp):
                self.logger.log(f"Abort moving to {detour_wp.tolist()} — segment crosses original obstacle")
                crossed_blocked = True
                break

            u = -self.k * (x - detour_wp)
            u = np.clip(u, -1, 1)
            x = x + u * self.dt

            self.logger.log(f"Scan step begin at {x.tolist()}")
            hits, hit_flag, _ = self._scan_and_detect(x)

            self.logger.log(f"Scan step end: obstacle_detected={hit_flag}")

            self.positions.append(x.copy())
            self.lidar_data.append((x.copy(), hits))

            # >>> add this to feed the heatmap/clusters for every in-motion scan <<<
            self.mem.add_scan(x, hits, self.range_limit)

            if hit_flag:
                lidar_blocked = True

            dist = np.linalg.norm(x - detour_wp)
            if last_dist - dist < 1e-4:
                no_progress_count += 1
            else:
                no_progress_count = 0
            last_dist = dist

            if self._goal_visible(x):
                self.logger.log("Goal visible during detour step; switching to straight navigation")
                break

            if no_progress_count >= max_no_progress:
                break

        if np.linalg.norm(x - detour_wp) <= self.threshold:
            self.logger.log(f"Reached waypoint {detour_wp.tolist() if hasattr(detour_wp, 'tolist') else list(detour_wp)}")
            return True

        if crossed_blocked:
            return False
        if lidar_blocked:
            self.logger.log(f"Could not reach waypoint {detour_wp.tolist() if hasattr(detour_wp, 'tolist') else list(detour_wp)} — blocked by nearby obstacle (LIDAR hit).")
            return False
        if no_progress_count >= max_no_progress:
            self.logger.log(f"Could not reach waypoint {detour_wp.tolist() if hasattr(detour_wp, 'tolist') else list(detour_wp)} — no progress (stuck).")
            return False
        if guard >= 600:
            self.logger.log(f"Could not reach waypoint {detour_wp.tolist() if hasattr(detour_wp, 'tolist') else list(detour_wp)} — timeout.")
            return False

        self.logger.log(f"Could not reach waypoint {detour_wp.tolist() if hasattr(detour_wp, 'tolist') else list(detour_wp)} — controller stop.")
        return False

    # ======================= ANIMATION =======================
    # ======================= ANIMATION =======================
    def _create_animation(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.N)
        ax.set_ylim(0, self.N)

        ax.set_aspect('equal')
        #ax.set_title("Navigation with LIDAR (  detours; finish chain before goal)")

        # draw original obstacles
        for obs in self.obstacles:
            rect = patches.Polygon(obs, closed=True, color='black')
            ax.add_patch(rect)

        agent_dot, = ax.plot([], [], 'go', markersize=8)
        ax.plot(self.goal[0], self.goal[1], 'b*', markersize=12)
        path_line, = ax.plot([], [], 'k-', linewidth=1.5)

        # NEW: text overlay for coordinates
        coord_text = ax.text(
            0.5, 1.02, "", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10, color="blue"
        )

        lidar_lines = []
        collision_markers = []

        def init():
            agent_dot.set_data([], [])
            path_line.set_data([], [])
            coord_text.set_text("")
            return [agent_dot, path_line, coord_text]

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

            # update coordinate overlay
            coord_text.set_text(f"Agent at ({pos[0]:.2f}, {pos[1]:.2f})")

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


            return [agent_dot, path_line, coord_text] + lidar_lines + collision_markers

        ani = FuncAnimation(fig, update, frames=len(self.lidar_data),
                            init_func=init, blit=False, interval=80)

        writer = FFMpegWriter(fps=12)
        ani.save(self.video_path, writer=writer)
        plt.close()



if __name__ == "__main__":
    nav = NavigationController()
    nav.run()