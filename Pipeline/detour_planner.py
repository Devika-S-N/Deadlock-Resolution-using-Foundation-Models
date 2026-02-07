import numpy as np
import random
from shapely.geometry import Point, LineString
from sklearn.decomposition import PCA

_last_tangent = None   # remembers last wall direction
_corner_hold = 0       # small hysteresis to avoid rapid tangent flips
_last_temp_goal = None
_fail_count = 0


def find_safe_temp_goal(robot_pos, wall_normal, lidar_hits, offset=1.0, max_tries=5):
    """
    Finds a temporary goal progressively farther away from the obstacle,
    ensuring it’s not too close to any LIDAR hit points.
    """
    import numpy as np

    lidar_pts = np.array(lidar_hits)
    best_goal = None

    for i in range(1, max_tries + 1):
        trial_goal = robot_pos + wall_normal * (offset * i * 1.5)
        dists = np.linalg.norm(lidar_pts - trial_goal, axis=1)
        if np.all(dists > 0.5):  # 0.5m clearance from any hit
            best_goal = trial_goal
            print(f" Safe temporary goal set at {np.round(best_goal, 2)} after {i} tries")
            break

    if best_goal is None:
        best_goal = robot_pos + wall_normal * (offset * max_tries * 2)
        print(f" Couldn’t find fully clear temp goal — using fallback at {np.round(best_goal, 2)}")

    return best_goal


def get_detour_waypoints(start, goal, lidar_hits,
                         step_size=0.5,
                         max_iter=300,
                         goal_sample_rate=0.2,
                         map_bounds=(0, 15, 0, 15),
                         open_gap_thresh=1.2,
                         tangent_change_thresh=np.deg2rad(45)):
    """
    Hybrid local detour planner using:
      1. Dynamic wall-segment re-estimation (adaptive tangent logic)
      2. Local doorway/open-space detection
      3. Short-range RRT toward a temporary goal
    """

    global _last_tangent, _corner_hold

    # ---------------------------------------------------------------------
    #   Handle trivial case (no obstacle hits)
    # ---------------------------------------------------------------------
    if not lidar_hits:
        print(" No LIDAR hits — direct line to global goal.")
        return [goal]

    pts = np.array([p for _, p in lidar_hits])
    start = np.array(start)
    goal = np.array(goal)

    # ---------------------------------------------------------------------
    #  Estimate current wall tangent via PCA
    # ---------------------------------------------------------------------
    cx, cy = np.mean(pts, axis=0)
    centered = pts - [cx, cy]
    if len(pts) >= 2:
        pca = PCA(n_components=2)
        pca.fit(centered)
        tangent = pca.components_[0] / np.linalg.norm(pca.components_[0])
    else:
        tangent = np.array([1.0, 0.0])

    normal = np.array([-tangent[1], tangent[0]])
    to_robot = np.array(start) - [cx, cy]
    if np.dot(to_robot, normal) < 0:
        normal = -normal

    # ---------------------------------------------------------------------
    #  Adaptive tangent switching (corner detection)
    # ---------------------------------------------------------------------
    if _last_tangent is not None:
        angle = np.arccos(np.clip(np.dot(_last_tangent, tangent), -1, 1))
        if angle > tangent_change_thresh and _corner_hold == 0:
            print(f" Corner detected ({np.degrees(angle):.1f}°). Updating wall direction.")
            _corner_hold = 5   # hold for few frames
            _last_tangent = tangent
        elif _corner_hold > 0:
            _corner_hold -= 1
        else:
            _last_tangent = tangent
    else:
        _last_tangent = tangent

    # ---------------------------------------------------------------------
    #  Detect open-space / doorway direction
    # ---------------------------------------------------------------------
    # Sort hit points by angle around robot
    rel = pts - start
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    sorted_idx = np.argsort(angles)
    gaps = np.diff(angles[sorted_idx])

    # Find biggest angular gap between consecutive hits
    max_gap = np.max(gaps) if len(gaps) > 0 else 0
    if max_gap > np.deg2rad(20):  # some open direction
        max_gap_idx = np.argmax(gaps)
        a1 = angles[sorted_idx][max_gap_idx]
        a2 = angles[sorted_idx][max_gap_idx + 1]
        open_angle = (a1 + a2) / 2.0
        open_dir = np.array([np.cos(open_angle), np.sin(open_angle)])
        print(" Possible doorway/open space detected.")
    else:
        open_dir = None
    # ---------------------------------------------------------------------
    #  Compute tangent-aligned temporary goal (wall-following)
    # ---------------------------------------------------------------------
    global _last_temp_goal, _fail_count

    step_dist = 1.0

    # Determine side of wall relative to robot
    to_robot_vec = start - np.array([cx, cy])
    side_sign = np.sign(np.dot(to_robot_vec, normal))
    safe_normal = normal * side_sign

    # Default: move parallel to wall tangent
    base_dir = tangent

    # If a large corner is detected, follow new tangent immediately
    if _last_tangent is not None:
        angle_diff = np.degrees(np.arccos(np.clip(np.dot(_last_tangent, tangent), -1, 1)))
        if angle_diff > 60:
            print(f" Sharp corner detected ({angle_diff:.1f}°) — updating tangent.")
            base_dir = tangent
        else:
            base_dir = _last_tangent

    # Place temporary goal parallel to wall, offset slightly away from it
    # temp_goal = start + base_dir * step_dist + safe_normal * 0.4
    # ---------------------------------------------------------------------
    #  Goal-biased tangent-aligned temporary goal
    # ---------------------------------------------------------------------
    goal_vec = goal - start
    goal_vec /= np.linalg.norm(goal_vec)

    # Project goal direction onto tangent
    proj_len = np.dot(goal_vec, base_dir)
    goal_along_tangent = base_dir * np.sign(proj_len)  # ensure same half-space as goal

    # Combine tangent motion + normal clearance
    step_dist = 1.0
    temp_goal = start + goal_along_tangent * step_dist + safe_normal * 0.4

    # Clamp within bounds
    xmin, xmax, ymin, ymax = map_bounds
    temp_goal[0] = np.clip(temp_goal[0], xmin + 0.5, xmax - 0.5)
    temp_goal[1] = np.clip(temp_goal[1], ymin + 0.5, ymax - 0.5)

    print(f" Goal-biased tangent temporary goal -> {np.round(temp_goal, 2)}")


    # Ensure the temporary goal remains within bounds
    xmin, xmax, ymin, ymax = map_bounds
    temp_goal[0] = np.clip(temp_goal[0], xmin + 0.5, xmax - 0.5)
    temp_goal[1] = np.clip(temp_goal[1], ymin + 0.5, ymax - 0.5)

    print(f" Tangent-aligned temporary goal -> {np.round(temp_goal, 2)} (parallel to wall)")
    _last_temp_goal = temp_goal




    # ---------------------------------------------------------------------
    #   Run short RRT to reach temporary goal
    # ---------------------------------------------------------------------
    waypoints = _rrt_path(start, temp_goal, pts,
                        step_size=step_size,
                        max_iter=max_iter,
                        goal_sample_rate=goal_sample_rate,
                        bounds=map_bounds)

    # ---------------------------------------------------------------------
    #  Debug inspection (interactive)
    # ---------------------------------------------------------------------
    print("\n RRT returned waypoints:")
    for i, p in enumerate(waypoints):
        print(f"   {i+1}. {np.round(p, 2)}")

    # optional pause for user inspection
    try:
        ans = input(" Proceed with these detour points? (y/n/r) ").strip().lower()
        if ans == "r":
            print(" Retrying RRT with same temporary goal...")
            waypoints = _rrt_path(start, temp_goal, pts,
                                step_size=step_size,
                                max_iter=max_iter*2,        # double attempts
                                goal_sample_rate=goal_sample_rate,
                                bounds=map_bounds)
        elif ans != "y":
            print(" Skipping this detour. No motion will occur.")
            waypoints = []
    except EOFError:
        # non-interactive run (e.g., redirected output)
        pass
    return waypoints


# -------------------------------------------------------------------------
# RRT implementation (simplified)
# -------------------------------------------------------------------------
def _rrt_path(start, goal, obs,
              step_size=0.5,
              max_iter=300,
              goal_sample_rate=0.2,
              bounds=(0, 15, 0, 15)):

    class Node:
        def __init__(self, pos, parent=None):
            self.pos = np.array(pos)
            self.parent = parent

    def collision(p1, p2):
        line = LineString([p1, p2])
        for pt in obs:
            if line.distance(Point(pt)) < 0.25:
                return True
        return False

    min_x, max_x, min_y, max_y = bounds
    start, goal = np.array(start), np.array(goal)
    nodes = [Node(start)]

    for i in range(max_iter):
        if random.random() < goal_sample_rate:
            rnd = goal
        else:
            rnd = np.array([
                random.uniform(min_x, max_x),
                random.uniform(min_y, max_y)
            ])

        dists = [np.linalg.norm(n.pos - rnd) for n in nodes]
        nearest = nodes[np.argmin(dists)]
        direction = rnd - nearest.pos
        if np.linalg.norm(direction) < 1e-6:
            continue
        direction /= np.linalg.norm(direction)
        new_pos = nearest.pos + direction * step_size

        if collision(nearest.pos, new_pos):
            continue

        new_node = Node(new_pos, nearest)
        nodes.append(new_node)

        if np.linalg.norm(new_pos - goal) < step_size:
            print(f" RRT reached temporary goal in {i+1} iterations.")
            return _extract_path(Node(goal, new_node))

    print(" RRT failed to reach temporary goal — using nearest partial path.")
    nearest_to_goal = min(nodes, key=lambda n: np.linalg.norm(n.pos - goal))
    return _extract_path(nearest_to_goal)


def _extract_path(node):
    """Backtrack path from given node."""
    path = []
    while node is not None:
        path.append(tuple(node.pos))
        node = node.parent
    path.reverse()
    return path
