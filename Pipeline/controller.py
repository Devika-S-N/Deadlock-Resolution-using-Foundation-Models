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
from VLM import query_detour_waypoints
from memory import Memory
import signal
import sys


def handle_exit(signum, frame):
    print("\nInterrupted — saving current plot before exit...")
    try:
        plt.savefig(os.path.join(nav.run_dir, "interrupted_frame.png"))
        print(f"Saved current figure to {os.path.join(nav.run_dir, 'interrupted_frame.png')}")
        plt.close('all')
    except Exception as e:
        print(f"Could not save figure: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)   # Ctrl + C
signal.signal(signal.SIGTERM, handle_exit)  # system terminate


class NavigationController:
    def __init__(self):
        # --- environment ---
        self.grid_size = 15
        self.num_rooms = 3
        self.num_obstacles = 10
        self.seed = None
        self.difficulty = "medium"

        # --- logger (create once, share everywhere) ---
        base_log_dir = "logs"
        os.makedirs(base_log_dir, exist_ok=True)

        run_id = f"run_{self.logger.timestamp}" if hasattr(self, "logger") else None
        # temporary logger to get timestamp
        _tmp_logger = Logger(log_dir=base_log_dir, strict_order=True)
        run_id = f"run_{_tmp_logger.timestamp}"
        _tmp_logger.close()

        self.run_dir = os.path.join(base_log_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # --- logger (NOW inside run_dir) ---
        self.logger = Logger(log_dir=self.run_dir, strict_order=True)
        self.logger.log(f"Run directory created at {self.run_dir}")

        self.env = Environment(
            N=self.grid_size,
            num_rooms=self.num_rooms,
            num_obstacles=self.num_obstacles,
            seed=self.seed,
            out_dir=self.run_dir,     
            difficulty=self.difficulty
        )

        self.grid, self.agent_start, self.goal, self.obstacles = self.env.create_environment()
        # grid is (N,N). Use that N for all plots
        self.N = int(self.grid.shape[0])
        self.logger.log(f"Environment initialized with agent at {self.agent_start} and goal at {self.goal}")

        # --- lidar (use the SAME logger) ---
        self.lidar = LidarObstacleDetector(
            self.grid, self.obstacles,
            resolution=0.1, range_limit=1.0, angle_increment=np.deg2rad(10),
            logger=self.logger
        )
        self.range_limit = 1.0 # keep in sync with lidar

        # --- motion ---
        self.dt = 0.05
        self.k = 2.0
        self.threshold = 0.1

        # --- geometry: original + buffered (for prompt only) ---
        self.buffer = 0.3  # used for what we SEND to the  
        self.polys = [Polygon(pts) for pts in self.obstacles]  # ORIGINAL polys (validation)
        self.buffered_polys = [poly.buffer(self.buffer, join_style=2) for poly in self.polys]  # PROMPT ONLY

        # --- trace & logs ---
        self.positions = [np.array(self.agent_start, dtype=float).copy()]
        self.lidar_data = []
        self.visited_waypoints = []

        self.log_dir = self.run_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.video_path = os.path.join(self.log_dir, f"navigation_{self.logger.timestamp}.mp4")

        # --- memory initialisation ---
        self.mem = Memory(bounds=(0, self.N, 0, self.N), cell=0.05, R=0.8, live_viz=True, record=True, log_dir=self.log_dir, run_id=self.logger.timestamp)
        self.mem.set_goal(self.goal[0], self.goal[1])  


        self.failed_detour_sets = []




    # ======================= MAIN LOOP =======================
    def run(self):
        x = np.array(self.agent_start, dtype=float)
        print(f"Starting navigation from {x.tolist()} to goal at {list(self.goal)}")
        self.logger.log(f"Starting navigation from {x.tolist()} to goal at {list(self.goal)}")

        step_guard = 0
        max_steps = 2000

        try:
            while np.linalg.norm(x - self.goal) > self.threshold and step_guard < max_steps:
                step_guard += 1

                # --- scan first: bracket with begin/end so ordering is obvious ---
                self.logger.log(f"Scan step begin at {x.tolist()}")
                hits, obstacle_detected, red_hits = self._scan_and_detect(x)

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



                target = self.goal
                u = -self.k * (x - self.goal)
                print(" Moving toward global goal")
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
        except KeyboardInterrupt:
            handle_exit(None, None)


    # ======================= LLM DETOUR REPLACEMENT =======================
    def _llm_only_detour(self, x, min_attempts=1):
        """
        Local detour planner using only LIDAR hit points and a VLM.
        Retries VLM calls if returned waypoints are invalid or unreachable.
        """
        max_attempts = 5
        attempt = 0
        retry_note = None



        self.logger.log("===== DETOUR TRIGGERED =====")
        self.logger.log(f"Agent position: {x.tolist()}")
        self.logger.log(f"Goal position: {list(self.goal)}")

        # Perform a fresh scan at current position
        hits, _, red_hits = self._scan_and_detect(x)

        self.logger.log(f"LIDAR hit count: {len(red_hits)}")
        self.logger.log(f"LIDAR hit points: {np.round(red_hits, 3).tolist()}")

        # Capture clean snapshot for VLM
        image_path = self.mem.save_clean_snapshot(
            agent=x,
            goal=self.goal,
            lidar_hits=red_hits
        )
        self.logger.log(f"VLM snapshot saved at: {image_path}")

        model_used = "gpt-4o"
        self.logger.log(f"Querying VLM model: {model_used}")

        from VLM import query_detour_waypoints

        # ---------------- RETRY LOOP ----------------
        while attempt < max_attempts:
            attempt += 1

            if attempt > 1:
                self.logger.log(
                    f"Retrying VLM detour (attempt {attempt}/{max_attempts}) "
                    f"due to invalid or unreachable waypoints"
                )

            self.logger.log(f"Calling VLM for detour (attempt {attempt})")

            if self.failed_detour_sets:
                retry_note = (
                    "The following waypoint sets were INVALID and caused collision or deadlock:\n"
                    f"{self.failed_detour_sets}\n\n"
                    "DO NOT repeat or closely approximate these points. "
                    "Choose a clearly different direction and a different side of the obstacle."
                )

            detour_points = query_detour_waypoints(
                image_path=image_path,
                agent=tuple(x),
                goal=tuple(self.goal),
                lidar_hits=[tuple(p) for p in red_hits],
                model=model_used,
                retry_note=retry_note
            )



            if detour_points:
                self.logger.log(
                    f"VLM returned {len(detour_points)} waypoints: "
                    f"{[list(map(lambda v: round(v, 3), wp)) for wp in detour_points]}"
                )
            else:
                self.logger.log("VLM returned NO waypoints")
                retry_note = "No waypoints were returned. Choose a different local path."
                continue

            # -------- VALIDATION AGAINST ORIGINAL OBSTACLES --------
            valid_chain = self._validate_against_original(x, detour_points)

            if valid_chain:
                self.logger.log(
                    f"Detour validation result: SUCCESS "
                    f"({len(valid_chain)} waypoints accepted)"
                )
                # --- add yellow detour waypoints to live heatmap ---
                self.mem.add_detour_points(valid_chain)
                self.logger.log(f"Added {len(valid_chain)} detour waypoints to live heatmap (yellow markers).")

                self.logger.log("===== DETOUR END =====")
                return valid_chain

            # -------- VALIDATION FAILED --------
            self.logger.log(
                "Detour validation result: FAILED — "
                "waypoints intersect obstacles or are unreachable"
            )
            # remember failed attempt so retries avoid it
            self.failed_detour_sets.append(
                [tuple(map(float, wp)) for wp in detour_points]
            )


            retry_note = (
                "Previous waypoints were invalid or collided with obstacles. "
                "Avoid those regions and provide a different local detour."
            )

        # ---------------- RETRIES EXHAUSTED ----------------
        self.logger.log(
            "VLM detour FAILED after maximum retries. "
            "Stopping navigation safely."
        )
        self.logger.log("===== DETOUR END =====")

        return []





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

    def _waypoint_blocked(self, waypoint, current_pos, tolerance=0.3):
        """
        Returns True if the waypoint lies within the newly detected obstacle region.
        """
        hits = self.lidar.scan(current_pos)
        obstacle_points = [end for _, end in hits]

        if not obstacle_points:
            return False

        for obs in obstacle_points:
            if np.linalg.norm(np.array(obs) - np.array(waypoint)) < tolerance:
                self.logger.log(f"Waypoint {np.round(waypoint, 2)} is blocked by new obstacle.")
                return True
        return False




    # ======================= OTHER HELPERS =======================

    def _scan_and_detect(self, x):
        hits = self.lidar.scan(x)
        obstacle_detected = len(hits) > 0
        red_hits = [end for _, end in hits]

        if self.logger:
            self.logger.log(f"LIDAR detected {len(red_hits)} obstacle hits: {np.round(red_hits, 3).tolist()}")

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

            # --- proactive waypoint blockage check (during traversal) ---
            if self._waypoint_blocked(detour_wp, x, tolerance=0.3):
                self.logger.log(f"Waypoint {np.round(detour_wp, 2)} became blocked by new obstacle during approach. Replanning.")
                # Immediately trigger requery
                new_detour = self._llm_only_detour(x)
                if new_detour:
                    # follow new detour points directly
                    for wp_new in new_detour:
                        if np.linalg.norm(x - self.goal) <= self.threshold:
                            break
                        self._go_to_waypoint_with_checks(wp_new)
                        x = self.positions[-1].copy()
                else:
                    self.logger.log("Replanning failed — stopping movement.")
                return False


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
                # All rays here are hits, so just mark them red
                color = 'r'
                line = ax.plot(
                    [pos[0], end[0]],
                    [pos[1], end[1]],
                    color=color,
                    linewidth=1
                )[0]
                lidar_lines.append(line)
                # Mark hit point
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