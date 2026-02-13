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
    def __init__(self, different=False):
        # --- environment ---
        self.grid_size = 15
        self.num_rooms = 3
        self.num_obstacles = 10
        self.seed = None
        self.difficulty = "medium"
        # reproducibility control
        self.different = different
        self.fixed_seed = 51   # <-- any constant number
        self.seed = None if self.different else self.fixed_seed

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
        # --- split geometry coming from environment.py ---
        self.obstacle_pts = [o for o in self.obstacles if isinstance(o, list)]          # rectangle obstacles (list of pts)
        self.wall_lines   = [o for o in self.obstacles if isinstance(o, LineString)]   # room/boundary walls (LineString)

        # Polygons for "filled" obstacles only (used for contains checks)
        self.polys = [Polygon(pts) for pts in self.obstacle_pts]

        # Everything the LIDAR should hit = polygons + wall lines
        self.obstacles_geom = self.polys + self.wall_lines

        # grid is (N,N). Use that N for all plots
        self.N = int(self.grid.shape[0])
        self.logger.log(f"Environment initialized with agent at {self.agent_start} and goal at {self.goal}")

        # --- lidar (use the SAME logger) ---
        self.lidar = LidarObstacleDetector(self.grid, self.obstacles_geom,
            resolution=0.1, range_limit=1.0, angle_increment=np.deg2rad(10),
            logger=self.logger
        )
        self.range_limit = 1.0 # keep in sync with lidar

        # --- motion ---
        self.dt = 0.05
        self.k = 2.0
        self.threshold = 0.4  # Goal reached threshold - realistic for navigation with obstacles

        # --- geometry: original + buffered (for prompt only) ---
        self.buffer = 0.3  # used for what we SEND to the  
        # self.polys = [Polygon(pts) for pts in self.obstacles]  # ORIGINAL polys (validation)
        # Obstacles are already Shapely geometries (Polygon or LineString)

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
    def validate_waypoint_progress(self, agent, goal, proposed_waypoints, retry_count=0):
        """
        SOLUTION 5: Validate that waypoints make progress toward goal.
        
        Uses ADAPTIVE tolerance that increases with retry attempts.
        This allows more exploration when agent is stuck.
        Returns (is_valid: bool, message: str)
        """
        if not proposed_waypoints:
            return False, "No waypoints provided"
        
        current_dist = np.linalg.norm(np.array(goal) - np.array(agent))
        
        # ADAPTIVE TOLERANCE: Becomes more lenient with each retry
        # Retry 1: 10% tolerance
        # Retry 2: 20% tolerance  
        # Retry 3: 30% tolerance
        # Retry 4: 40% tolerance
        # Retry 5: 50% tolerance (very lenient - try anything!)
        tolerance_multiplier = 1.0 + (retry_count * 0.10)  # 1.1, 1.2, 1.3, 1.4, 1.5
        
        # DIRECTIONAL PROGRESS: Check if making progress in X OR Y
        # This is crucial for rectangular rooms with doors
        dx_to_goal = abs(goal[0] - agent[0])
        dy_to_goal = abs(goal[1] - agent[1])
        
        for i, wp in enumerate(proposed_waypoints):
            wp_array = np.array(wp)
            
            # Check distance progress (with adaptive tolerance)
            wp_dist = np.linalg.norm(np.array(goal) - wp_array)
            
            # Allow first 2 waypoints to move away (for obstacle avoidance)
            if i >= 2:
                # Check if within adaptive tolerance
                if wp_dist > current_dist * tolerance_multiplier:
                    # ALSO check directional progress as backup
                    dx_wp = abs(goal[0] - wp[0])
                    dy_wp = abs(goal[1] - wp[1])
                    
                    # If getting closer in EITHER direction, allow it
                    x_progress = dx_wp < dx_to_goal * tolerance_multiplier
                    y_progress = dy_wp < dy_to_goal * tolerance_multiplier
                    
                    if not (x_progress or y_progress):
                        return False, f"Waypoint {i+1} at {wp} doesn't progress (dist {wp_dist:.2f} > {current_dist:.2f}, tolerance={tolerance_multiplier:.1f}x)"
        
        # Check final waypoint makes SOME progress (even with high retry)
        final_wp_dist = np.linalg.norm(np.array(goal) - np.array(proposed_waypoints[-1]))
        max_final_tolerance = 1.0 + (retry_count * 0.15)  # More lenient: 1.15, 1.30, 1.45, 1.60, 1.75
        
        if final_wp_dist > current_dist * max_final_tolerance:
            return False, f"Final waypoint too far from goal (dist {final_wp_dist:.2f} vs current {current_dist:.2f}, max_tolerance={max_final_tolerance:.1f}x)"
        
        return True, f"Waypoints make acceptable progress (tolerance={tolerance_multiplier:.1f}x, retry={retry_count})"

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
                    for i, detour_wp in enumerate(detour_chain):
                        if np.linalg.norm(x - self.goal) <= self.threshold:
                            break
                        self._go_to_waypoint_with_checks(
                            detour_wp,
                            waypoint_index=i,
                            total_waypoints=len(detour_chain)
                        )
                        x = self.positions[-1].copy()
                    continue
                    # Detour completed → archive it
                    self.mem.archive_current_detours()




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
        # Convert all existing detours to past (yellow) before re-query
        self.mem.archive_current_detours()

        # Get all yellow waypoint history from memory
        all_yellow_waypoints = self.mem._past_detour_points  # list of [x, y]
        self.logger.log(f"Total yellow waypoints in history: {len(all_yellow_waypoints)}")

        # Capture clean snapshot for VLM
        # Explicitly get accumulated hits
        all_accumulated_hits = self.mem.get_accumulated_lidar_hits()
        self.logger.log(f"Accumulated LIDAR hits: {len(all_accumulated_hits)}")

        image_path = self.mem.save_clean_snapshot(
            agent=x,
            goal=self.goal,
            lidar_hits=all_accumulated_hits  # Now uses ALL hits!
        )
        self.logger.log(f"VLM snapshot saved at: {image_path}")

        model_used = "gpt-5.2"
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
                yellow_waypoints=all_yellow_waypoints,  # NEW: pass yellow waypoint history
                model=model_used,
                retry_note=retry_note,
                logger=self.logger  # NEW: Pass logger for prompt logging
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

            # -------- YELLOW WAYPOINT OSCILLATION PREVENTION --------
            # Validate waypoints are not too close to previous yellow waypoints
            all_yellow = all_yellow_waypoints  # Already retrieved earlier in this function
            yellow_valid, yellow_msg, filtered_waypoints = self.validate_yellow_waypoint_distance(
                detour_points,
                all_yellow,
                min_distance=0.7  # Must be 0.7+ units from any yellow waypoint
            )
            
            if not yellow_valid:
                self.logger.log(f"Yellow waypoint validation FAILED: {yellow_msg}")
                retry_note = (
                    f"Previous waypoints were placed too close to already-visited locations. {yellow_msg}. "
                    f"You MUST generate waypoints in a COMPLETELY DIFFERENT area, at least 1.5 units away "
                    f"from recent waypoints: {all_yellow[-5:] if len(all_yellow) > 5 else all_yellow}"
                )
                continue
            
            if len(filtered_waypoints) < len(detour_points):
                self.logger.log(f"Yellow waypoint validation PARTIAL: {yellow_msg}")
                detour_points = filtered_waypoints  # Use filtered set
                if len(detour_points) == 0:
                    retry_note = f"All waypoints too close to yellow history. Generate waypoints in a NEW area."
                    continue
            else:
                self.logger.log(f"Yellow waypoint validation PASSED: {yellow_msg}")

            # -------- SOLUTION 5: PROGRESS VALIDATION --------
            # Pass retry count for adaptive tolerance (attempt starts at 1, so subtract 1 for 0-indexed)
            progress_valid, progress_msg = self.validate_waypoint_progress(x, self.goal, detour_points, retry_count=(attempt - 1))
            if not progress_valid:
                self.logger.log(f"Progress validation FAILED: {progress_msg}")
                retry_note = f"Previous waypoints didn't make progress toward goal. {progress_msg}. Generate waypoints that get closer to the goal."
                continue
            else:
                self.logger.log(f"Progress validation PASSED: {progress_msg}")

            # -------- VALIDATION AGAINST ORIGINAL OBSTACLES --------
            valid_chain = self._validate_against_original(x, detour_points)

            if valid_chain:
                self.logger.log(
                    f"Detour validation result: SUCCESS "
                    f"({len(valid_chain)} waypoints accepted)"
                )
                # --- add yellow detour waypoints to live heatmap ---
                self.mem.add_current_detours(valid_chain)
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

        # ---------------- RETRIES EXHAUSTED - TRY ESCAPE HEURISTIC ----------------
        self.logger.log(
            "VLM detour FAILED after maximum retries. "
            "Attempting GEOMETRIC ESCAPE HEURISTIC as last resort."
        )
        
        # Try escape heuristic - generate waypoints based on obstacle geometry
        escape_wps = self._generate_escape_waypoints(
            x.tolist() if hasattr(x, 'tolist') else x,
            self.goal.tolist() if hasattr(self.goal, 'tolist') else self.goal,
            red_hits
        )
        
        if escape_wps:
            self.logger.log(f"Validating {len(escape_wps)} escape heuristic waypoints...")
            # Validate escape waypoints against original obstacles
            valid_escape = self._validate_against_original(x, escape_wps)
            
            if valid_escape and len(valid_escape) > 0:
                self.logger.log(
                    f"ESCAPE HEURISTIC SUCCESS! Using {len(valid_escape)} geometrically-generated waypoints"
                )
                # Add to memory like normal waypoints
                self.mem.add_current_detours(valid_escape)
                self.logger.log("===== DETOUR END (via escape heuristic) =====")
                return valid_escape
            else:
                self.logger.log("Escape heuristic waypoints also blocked by obstacles")
        else:
            self.logger.log("Could not generate escape waypoints")
        
        # If even escape heuristic fails, give up
        self.logger.log("VLM detour AND escape heuristic FAILED. Stopping navigation safely.")
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

        # filled obstacles (polygons)
        for poly in self.polys:
            if seg.intersects(poly) and not seg.touches(poly):
                return True

        # walls (lines): any intersection should block
        for wall in self.wall_lines:
            if seg.intersects(wall):
                return True

        return False


    def _is_near_goal(self, x):
        """
        Check if agent is close enough to goal to activate 'near-goal mode'.
        In this mode, agent ignores obstacles and makes final push to goal.
        """
        distance = np.linalg.norm(x - self.goal)
        near_goal_threshold = 0.6  # Within 0.6 units - close enough for final approach
        return distance <= near_goal_threshold


    def validate_yellow_waypoint_distance(self, proposed_waypoints, yellow_history, min_distance=1.0):
        """
        SOLUTION: Prevent oscillation by rejecting waypoints too close to yellow history.
        
        The VLM sometimes ignores the "stay away from yellow waypoints" rule,
        causing the agent to oscillate between the same 2-3 points endlessly.
        This validation step explicitly checks and rejects such waypoints.
        
        Returns: (is_valid: bool, message: str, filtered_waypoints: list)
        """
        if not yellow_history or len(yellow_history) == 0:
            return True, "No yellow history to check", proposed_waypoints
        
        filtered = []
        rejected_count = 0
        
        for i, wp in enumerate(proposed_waypoints):
            # Check distance to all yellow waypoints
            too_close = False
            closest_dist = float('inf')
            closest_yellow = None
            
            for yellow_wp in yellow_history:
                dist = np.linalg.norm(np.array(wp) - np.array(yellow_wp))
                if dist < closest_dist:
                    closest_dist = dist
                    closest_yellow = yellow_wp
                
                if dist < min_distance:
                    too_close = True
                    break
            
            if too_close:
                self.logger.log(
                    f"Reject WP{i+1} at {wp}: Too close to yellow waypoint {closest_yellow} "
                    f"(distance: {closest_dist:.2f} < {min_distance:.2f})"
                )
                rejected_count += 1
            else:
                filtered.append(wp)
        
        if len(filtered) == 0:
            return False, f"All {len(proposed_waypoints)} waypoints rejected - too close to yellow history", []
        
        if len(filtered) < len(proposed_waypoints):
            return True, f"Partial: {rejected_count} waypoints too close to yellow history", filtered
        
        return True, "All waypoints clear of yellow history", filtered


    def _detect_tunnel(self, agent_pos, lidar_hits):
        """
        SOLUTION: Detect if agent is in a narrow tunnel/corridor.
        
        Tunnels cause oscillation because escape heuristic generates waypoints
        along the tunnel instead of toward the exit.
        
        Returns: (is_tunnel: bool, tunnel_direction: str, tunnel_axis: float)
        """
        if not lidar_hits or len(lidar_hits) < 3:
            return False, None, None
        
        hit_array = np.array(lidar_hits)
        
        # Check variance to detect tunnel orientation
        x_variance = np.var(hit_array[:, 0])
        y_variance = np.var(hit_array[:, 1])
        
        self.logger.log(f"Tunnel check: x_var={x_variance:.3f}, y_var={y_variance:.3f}")
        
        # Horizontal tunnel: hits have similar y-coordinates (low y-variance)
        if y_variance < 0.3:
            y_hits = hit_array[:, 1]
            # Check if hits are on BOTH sides (tunnel walls above and below)
            hits_above = np.sum(y_hits > agent_pos[1])
            hits_below = np.sum(y_hits < agent_pos[1])
            
            if hits_above > 0 and hits_below > 0:
                tunnel_y = np.mean(y_hits)
                self.logger.log(f"TUNNEL DETECTED: Horizontal tunnel at y={tunnel_y:.2f}")
                return True, "horizontal", tunnel_y
        
        # Vertical tunnel: hits have similar x-coordinates (low x-variance)
        # In _detect_tunnel(), add tunnel width check:
        if x_variance < 0.3:
            x_hits = hit_array[:, 0]
            hits_left = x_hits[x_hits < agent_pos[0]]
            hits_right = x_hits[x_hits > agent_pos[0]]
            
            if len(hits_left) > 0 and len(hits_right) > 0:
                left_max = np.max(hits_left)
                right_min = np.min(hits_right)
                tunnel_width = right_min - left_max
                
                # Only detect tunnel if walls are actually close
                if tunnel_width < 2.0:  # Actual tunnel, not just scattered hits
                    tunnel_x = np.mean(x_hits)
                    return True, "vertical", tunnel_x
        
        return False, None, None


    def _generate_tunnel_escape(self, agent_pos, goal_pos, tunnel_direction, tunnel_axis):
        """
        SOLUTION: Generate waypoints to EXIT a tunnel, not oscillate within it.
        
        Strategy: 
        1. Keep waypoints in TUNNEL CENTER (not at agent's position near wall)
        2. Move toward the tunnel END that's closer to goal
        
        Parameters:
        - agent_pos: Current agent position
        - goal_pos: Goal position
        - tunnel_direction: "horizontal" or "vertical"
        - tunnel_axis: Center coordinate of tunnel (y for horizontal, x for vertical)
        """
        self.logger.log(f"TUNNEL ESCAPE: Generating exit waypoints for {tunnel_direction} tunnel")
        self.logger.log(f"Tunnel center at {'y' if tunnel_direction == 'horizontal' else 'x'}={tunnel_axis:.2f}")
        self.logger.log(f"Agent at {'y' if tunnel_direction == 'horizontal' else 'x'}={agent_pos[1] if tunnel_direction == 'horizontal' else agent_pos[0]:.2f}")
        
        escape_waypoints = []
        
        if tunnel_direction == "horizontal":
            # Horizontal tunnel - keep y at tunnel center, vary x
            goal_direction = "right" if goal_pos[0] > agent_pos[0] else "left"
            self.logger.log(f"Goal is to the {goal_direction} - exiting tunnel in that direction")
            self.logger.log(f"Using tunnel center y={tunnel_axis:.2f} instead of agent y={agent_pos[1]:.2f}")
            
            if goal_direction == "right":
                # Move right through tunnel toward exit, staying in center
                escape_waypoints = [
                    [agent_pos[0] + 2.0, tunnel_axis],  # Use tunnel center!
                    [agent_pos[0] + 4.5, tunnel_axis],
                    [agent_pos[0] + 7.0, tunnel_axis],
                    [goal_pos[0], tunnel_axis]  # Reach goal x-coordinate at tunnel center
                ]
            else:
                # Move left through tunnel toward exit, staying in center
                escape_waypoints = [
                    [agent_pos[0] - 2.0, tunnel_axis],
                    [agent_pos[0] - 4.5, tunnel_axis],
                    [agent_pos[0] - 7.0, tunnel_axis],
                    [goal_pos[0], tunnel_axis]
                ]
        
        elif tunnel_direction == "vertical":
            # Vertical tunnel - keep x at tunnel center, vary y
            goal_direction = "up" if goal_pos[1] > agent_pos[1] else "down"
            self.logger.log(f"Goal is {goal_direction} - exiting tunnel in that direction")
            self.logger.log(f"Using tunnel center x={tunnel_axis:.2f} instead of agent x={agent_pos[0]:.2f}")
            
            if goal_direction == "up":
                # Move up through tunnel toward exit, staying in center
                escape_waypoints = [
                    [tunnel_axis, agent_pos[1] + 2.0],
                    [tunnel_axis, agent_pos[1] + 4.5],
                    [tunnel_axis, agent_pos[1] + 7.0],
                    [tunnel_axis, goal_pos[1]]
                ]
            else:
                # Move down through tunnel toward exit, staying in center
                escape_waypoints = [
                    [tunnel_axis, agent_pos[1] - 2.0],
                    [tunnel_axis, agent_pos[1] - 4.5],
                    [tunnel_axis, agent_pos[1] - 7.0],
                    [tunnel_axis, goal_pos[1]]
                ]
                
        
        # Clip to bounds
        escape_waypoints = [
            [max(0.5, min(14.5, wp[0])), max(0.5, min(14.5, wp[1]))] 
            for wp in escape_waypoints
        ]
        
        self.logger.log(f"Tunnel escape waypoints: {escape_waypoints}")
        return escape_waypoints


    def _generate_escape_waypoints(self, agent_pos, goal_pos, lidar_hits):
        """
        SOLUTION: Generate waypoints using geometric heuristic when VLM repeatedly fails.
        
        When VLM fails 5 times in a row, it's likely stuck trying the same strategy.
        This generates waypoints based on obstacle geometry instead.
        
        Strategy: 
        1. FIRST check if in tunnel - use tunnel escape
        2. ELSE detect wall orientation, move perpendicular to escape, then resume toward goal.
        """
        self.logger.log("ESCAPE HEURISTIC: VLM failed - using geometric waypoint generation")
        
        if not lidar_hits or len(lidar_hits) == 0:
            self.logger.log("No LIDAR hits - cannot determine escape direction")
            return None
        
        # # PRIORITY 1: Check if in tunnel (most common oscillation cause)
        # is_tunnel, tunnel_dir, tunnel_axis = self._detect_tunnel(agent_pos, lidar_hits)
        
        # if is_tunnel:
        #     # Use tunnel-specific escape strategy
        #     # Pass tunnel_axis so waypoints stay in tunnel center
        #     return self._generate_tunnel_escape(agent_pos, goal_pos, tunnel_dir, tunnel_axis)
        
        # # PRIORITY 2: Not in tunnel - use wall detection
        # self.logger.log("Not in tunnel - using wall-based escape heuristic")
        
        # Analyze LIDAR hits to find wall direction
        hit_array = np.array(lidar_hits)
        
        # Check if hits form horizontal or vertical wall
        y_variance = np.var(hit_array[:, 1])
        x_variance = np.var(hit_array[:, 0])
        
        self.logger.log(f"Wall analysis: x_var={x_variance:.3f}, y_var={y_variance:.3f}")
        
        escape_waypoints = []
        
        if y_variance < 0.2:  # Horizontal wall (all hits have similar y)
            wall_y = np.mean(hit_array[:, 1])
            self.logger.log(f"Detected HORIZONTAL wall at y={wall_y:.2f}")
            
            if agent_pos[1] < wall_y:
                # Wall is above - escape downward, then go around
                self.logger.log("Escaping DOWNWARD from horizontal wall")
                escape_waypoints = [
                    [agent_pos[0], agent_pos[1] - 1.5],  # Move down 1.5 units
                    [agent_pos[0] + 2.5, agent_pos[1] - 1.0],  # Move diagonally
                    [goal_pos[0], agent_pos[1] - 0.5],  # Align with goal x
                    [goal_pos[0], goal_pos[1]]  # Move to goal
                ]
            else:
                # Wall is below - escape upward, then go around
                self.logger.log("Escaping UPWARD from horizontal wall")
                escape_waypoints = [
                    [agent_pos[0], agent_pos[1] + 1.5],
                    [agent_pos[0] + 2.5, agent_pos[1] + 1.0],
                    [goal_pos[0], agent_pos[1] + 0.5],
                    [goal_pos[0], goal_pos[1]]
                ]
        
        elif x_variance < 0.2:  # Vertical wall (all hits have similar x)
            wall_x = np.mean(hit_array[:, 0])
            self.logger.log(f"Detected VERTICAL wall at x={wall_x:.2f}")
            
            if agent_pos[0] < wall_x:
                # Wall is to the right - escape leftward, then go around
                self.logger.log("Escaping LEFT from vertical wall")
                escape_waypoints = [
                    [agent_pos[0] - 1.5, agent_pos[1]],
                    [agent_pos[0] - 1.0, agent_pos[1] + 2.5],
                    [agent_pos[0] - 0.5, goal_pos[1]],
                    [goal_pos[0], goal_pos[1]]
                ]
            else:
                # Wall is to the left - escape rightward, then go around
                self.logger.log("Escaping RIGHT from vertical wall")
                escape_waypoints = [
                    [agent_pos[0] + 1.5, agent_pos[1]],
                    [agent_pos[0] + 1.0, agent_pos[1] + 2.5],
                    [agent_pos[0] + 0.5, goal_pos[1]],
                    [goal_pos[0], goal_pos[1]]
                ]
        
        else:
            # Complex/diagonal obstacle - move perpendicular to goal direction
            self.logger.log("Detected COMPLEX obstacle - using perpendicular escape")
            to_goal = np.array(goal_pos) - np.array(agent_pos)
            perpendicular = np.array([-to_goal[1], to_goal[0]])  # 90 degree rotation
            if np.linalg.norm(perpendicular) > 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular) * 2.5
            
            escape_waypoints = [
                (agent_pos + perpendicular).tolist(),
                (agent_pos + perpendicular + to_goal * 0.4).tolist(),
                (agent_pos + to_goal * 0.7).tolist(),
                goal_pos
            ]
        
        # Clip to environment bounds [0, 15]
        escape_waypoints = [
            [max(0.5, min(14.5, wp[0])), max(0.5, min(14.5, wp[1]))] 
            for wp in escape_waypoints
        ]
        
        self.logger.log(f"Generated {len(escape_waypoints)} escape waypoints: {escape_waypoints}")
        return escape_waypoints


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

        for poly in self.polys:
            if seg.intersects(poly) and not seg.touches(poly):
                return False

        for wall in self.wall_lines:
            if seg.intersects(wall):
                return False

        return True


    def _go_to_waypoint_with_checks(self, detour_wp, waypoint_index=0, total_waypoints=1):
        """
        Navigate to a single detour waypoint, doing LIDAR at every microstep.
        
        SOLUTION: Added waypoint_index tracking to prevent premature goal visibility checks.
        
        Parameters:
        - detour_wp: Target waypoint
        - waypoint_index: Which waypoint in sequence (0, 1, 2, ...)
        - total_waypoints: Total waypoints in detour chain
        """
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

            # CRITICAL: Check if goal reached BEFORE obstacle detection
            # This prevents infinite loops when near goal with nearby obstacles
            if np.linalg.norm(x - self.goal) <= self.threshold:
                self.logger.log(f"Reached goal at {x.tolist()} (distance: {np.linalg.norm(x - self.goal):.3f})")
                self.positions.append(x.copy())
                return True

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
                    for j, wp_new in enumerate(new_detour):
                        if np.linalg.norm(x - self.goal) <= self.threshold:
                            break
                        self._go_to_waypoint_with_checks(
                            wp_new,
                            waypoint_index=j,
                            total_waypoints=len(new_detour)
                        )
                        x = self.positions[-1].copy()
                else:
                    self.logger.log("Replanning failed — stopping movement.")
                return False


            u = -self.k * (x - detour_wp)
            u = np.clip(u, -1, 1)
            x = x + u * self.dt

            # NEAR-GOAL MODE: If very close to goal, skip obstacle detection and push through
            if self._is_near_goal(x):
                self.logger.log(f"Near-goal mode: ignoring obstacles (dist to goal: {np.linalg.norm(x - self.goal):.3f})")
                self.positions.append(x.copy())
                # Skip obstacle detection, continue toward goal
                continue

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

            # SOLUTION: Only check goal visibility after progressing through detour
            # Checking too early causes agent to abandon detour and hit obstacles
            if waypoint_index >= 2 or waypoint_index >= total_waypoints - 1:
                # Only check after completing 2+ waypoints OR on last waypoint
                if self._goal_visible(x):
                    self.logger.log(
                        f"Goal visible after waypoint {waypoint_index+1}/{total_waypoints} "
                        f"- switching to straight navigation"
                    )
                    break
            # else: Don't check goal visibility yet - commit to detour!

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
        # draw filled obstacles
        for pts in self.obstacle_pts:
            ax.add_patch(patches.Polygon(pts, closed=True, color='black'))

        # draw walls as lines
        for w in self.wall_lines:
            xw, yw = w.xy
            ax.plot(xw, yw, 'k-', linewidth=2)


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
    nav = NavigationController(different=False)
    nav.run()