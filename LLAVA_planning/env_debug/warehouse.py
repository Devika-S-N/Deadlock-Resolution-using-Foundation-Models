# warehouse.py
import os
import time
import random
import argparse
import json
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont


# ---------------- utils ----------------
def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def try_load_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ---------------- main generator ----------------
def generate_env(
    N=10,
    num_rooms=3,
    num_obstacles=4,
    seed=None,
    out_dir="logs/room_envs",
    cell_px=70,
    wall_px=12,
    obstacle_h_range=(1, 2),
    obstacle_w_range=(1, 4),
    max_global_restarts=600,
    auto_grow=True,
    grow_step=3,
    max_N=60,
    add_workstations=False,
    difficulty="hard", 
):
    """
    Base generator + Workstations.

    Key behavior:
      - rooms with doors (walls)
      - optional workstation thin walls
      - obstacles (filled rectangles)
      - agent and goal inside rooms

    NEW:
      - exports a unified occupancy grid (walls + obstacles treated as obstacles)
        using a 2x resolution grid:
          cell centers at (2r+1, 2c+1)
          vertical walls at (2r+1, 2c)
          horizontal walls at (2r, 2c+1)
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed if seed is not None else random.randrange(10**9))

    # ---- BFS helpers ----
    def build_empty_walls(N_):
        vwall = np.zeros((N_ + 1, N_), dtype=bool)  # vwall[col, r]
        hwall = np.zeros((N_ + 1, N_), dtype=bool)  # hwall[row, c]
        return vwall, hwall

    def reachable_set(N_, start, blocked, vwall, hwall):
        if blocked[start]:
            return set()

        q = deque([start])
        seen = {start}

        def can_move(a, b):
            (r1, c1), (r2, c2) = a, b
            if not (0 <= r2 < N_ and 0 <= c2 < N_):
                return False
            if blocked[r2, c2]:
                return False

            # note: vwall indexed as vwall[col, r]
            if r1 == r2 and c2 == c1 + 1:
                return not vwall[c1 + 1, r1]
            if r1 == r2 and c2 == c1 - 1:
                return not vwall[c1, r1]
            if c1 == c2 and r2 == r1 + 1:
                return not hwall[r1 + 1, c1]
            if c1 == c2 and r2 == r1 - 1:
                return not hwall[r1, c1]
            return False

        while q:
            r, c = q.popleft()
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < N_ and 0 <= nc < N_ and nxt not in seen:
                    if can_move((r, c), nxt):
                        seen.add(nxt)
                        q.append(nxt)
        return seen

    # ---- drawing ----
    font = try_load_font(16)
    font_small = try_load_font(13)

    BLACK = (0, 0, 0)
    GRID = (230, 230, 230)

    def bfs_shortest_path(N_, start, goal, blocked, vwall, hwall):
        """Return list of (r,c) from start->goal inclusive, or [] if no path."""
        if blocked[start] or blocked[goal]:
            return []

        def can_move(a, b):
            (r1, c1), (r2, c2) = a, b
            if not (0 <= r2 < N_ and 0 <= c2 < N_):
                return False
            if blocked[r2, c2]:
                return False

            # right
            if r1 == r2 and c2 == c1 + 1:
                return not vwall[c1 + 1, r1]
            # left
            if r1 == r2 and c2 == c1 - 1:
                return not vwall[c1, r1]
            # down
            if c1 == c2 and r2 == r1 + 1:
                return not hwall[r1 + 1, c1]
            # up
            if c1 == c2 and r2 == r1 - 1:
                return not hwall[r1, c1]
            return False

        q = deque([start])
        parent = {start: None}

        while q:
            cur = q.popleft()
            if cur == goal:
                break

            r, c = cur
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nxt = (r + dr, c + dc)
                if 0 <= nxt[0] < N_ and 0 <= nxt[1] < N_ and nxt not in parent:
                    if can_move(cur, nxt):
                        parent[nxt] = cur
                        q.append(nxt)

        if goal not in parent:
            return []

        # reconstruct
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path


    # ---------------- one build attempt ----------------
    def build_once(N_, local_seed):
        local_rng = random.Random(local_seed)

        # room walls
        vwall_room, hwall_room = build_empty_walls(N_)
        # workstation walls (thin, separate)
        vwall_ws, hwall_ws = build_empty_walls(N_)

        blocked = np.zeros((N_, N_), dtype=bool)

        rooms = []
        room_doors = []
        door_edges = []




        def vwall_total():
            return np.logical_or(vwall_room, vwall_ws)

        def hwall_total():
            return np.logical_or(hwall_room, hwall_ws)

        # ---- wall setters (room) ----
        def add_vwall_room(col, r0, r1):
            for r in range(r0, r1):
                vwall_room[col, r] = True

        def add_hwall_room(row, c0, c1):
            for c in range(c0, c1):
                hwall_room[row, c] = True

        # ---------------- Difficulty handling ----------------
        diff = difficulty.lower()

        # EASY MODE: skip rooms entirely, only obstacles
        if diff == "easy":
            rooms = []
            room_doors = []
            room_interiors = []
            room_anchors = []

            # no walls except boundary
            add_hwall_room(0, 0, N_)
            add_hwall_room(N_, 0, N_)
            add_vwall_room(0, 0, N_)
            add_vwall_room(N_, 0, N_)

            # agent / goal placed randomly in free space
            free_cells = [(r, c) for r in range(1, N_ - 1) for c in range(1, N_ - 1)]
            agent = local_rng.choice(free_cells)
            goal = local_rng.choice([p for p in free_cells if p != agent])

            # proceed to obstacle placement section later
            skip_room_phase = True
        else:
            skip_room_phase = False

        def room_bbox_cells(r0, c0, h, w, pad=0):
            cells = set()
            for r in range(r0 - pad, r0 + h + pad):
                for c in range(c0 - pad, c0 + w + pad):
                    if 0 <= r < N_ and 0 <= c < N_:
                        cells.add((r, c))
            return cells

        def cell_in_any_room(rr, cc, room_list):
            for (r0, c0, h, w) in room_list:
                if r0 <= rr < r0 + h and c0 <= cc < c0 + w:
                    return True
            return False

        # ---- outer border (room walls) ----
        add_hwall_room(0, 0, N_)
        add_hwall_room(N_, 0, N_)
        add_vwall_room(0, 0, N_)
        add_vwall_room(N_, 0, N_)

        # ---- room size picking ----
        def pick_room_size():
            min_dim = 4 if N_ <= 15 else 5
            target_area = int(0.40 * (N_ * N_) / max(1, num_rooms))
            target_dim = int(max(min_dim, round(target_area ** 0.5)))
            max_dim = min(N_ - 4, target_dim + 2)
            max_dim = max(min_dim, max_dim)
            h = local_rng.randint(min_dim, max_dim)
            w = local_rng.randint(min_dim, max_dim)
            return h, w

        if not skip_room_phase:
            # ---- place rooms ----
            occupied_for_rooms = set()
            room_pad = 1 if num_rooms <= 4 else 0

            for _ in range(num_rooms):
                placed = False
                for _try in range(3500):
                    h, w = pick_room_size()
                    if N_ - h - 1 < 1 or N_ - w - 1 < 1:
                        continue

                    r0 = local_rng.randint(1, N_ - h - 1)
                    c0 = local_rng.randint(1, N_ - w - 1)

                    candidate = room_bbox_cells(r0, c0, h, w, pad=room_pad)
                    if candidate & occupied_for_rooms:
                        continue

                    top, bot = r0, r0 + h
                    left, right = c0, c0 + w

                    add_hwall_room(top, left, right)
                    add_hwall_room(bot, left, right)
                    add_vwall_room(left, top, bot)
                    add_vwall_room(right, top, bot)

                    sides = ["top", "bottom", "left", "right"]
                    local_rng.shuffle(sides)

                    door_ok = False
                    # ---- door selection logic ----
                    door_count = 1
                    if diff == "medium":
                        door_count = local_rng.choice([1, 2])  # 1 or 2 doors

                    door_positions = []

                    for _door_i in range(door_count):
                        sides = ["top", "bottom", "left", "right"]
                        local_rng.shuffle(sides)
                        door_ok = False

                        for side in sides:
                            if side in ["top", "bottom"]:
                                if (right - left) < 3:
                                    continue
                                dc = local_rng.randint(left + 1, right - 2)
                                row = top if side == "top" else bot
                                outside_cell = (row - 1, dc) if side == "top" else (row, dc)
                                orr, occ = outside_cell
                                if not (0 <= orr < N_ and 0 <= occ < N_):
                                    continue

                                # in HARD we forbid opening into another room
                                if diff == "hard" and cell_in_any_room(orr, occ, rooms + [(r0, c0, h, w)]):
                                    continue

                                door_positions.append(("h", row, dc))
                                door_ok = True
                                break

                            else:  # left/right
                                if (bot - top) < 3:
                                    continue
                                dr = local_rng.randint(top + 1, bot - 2)
                                col = left if side == "left" else right
                                outside_cell = (dr, col - 1) if side == "left" else (dr, col)
                                orr, occ = outside_cell
                                if not (0 <= orr < N_ and 0 <= occ < N_):
                                    continue

                                if diff == "hard" and cell_in_any_room(orr, occ, rooms + [(r0, c0, h, w)]):
                                    continue

                                door_positions.append(("v", col, dr))
                                door_ok = True
                                break

                        if not door_ok and diff != "easy":
                            # rollback only if we are in medium/hard
                            return None

                    # register doors for this room
                    for dpos in door_positions:
                        room_doors.append(dpos)


                    if not door_ok:
                        return None

                    rooms.append((r0, c0, h, w))
                    occupied_for_rooms |= candidate
                    placed = True
                    break

                if not placed:
                    return None

            
            # ---- carve doors in room walls (WIDER DOORS: width = 3) ----
            DOOR_HALF_WIDTH = 1  # total width = 2*1 + 1 = 3

            for kind, a, b in room_doors:
                if kind == "h":
                    row, c = a, b
                    for dc in range(-DOOR_HALF_WIDTH, DOOR_HALF_WIDTH + 1):
                        cc = c + dc
                        if 0 <= cc < N_:
                            hwall_room[row, cc] = False
                else:
                    col, r = a, b
                    for dr in range(-DOOR_HALF_WIDTH, DOOR_HALF_WIDTH + 1):
                        rr = r + dr
                        if 0 <= rr < N_:
                            vwall_room[col, rr] = False


            # ---- build door_edges ----
            for kind, a, b in room_doors:
                if kind == "h":
                    row, c = a, b
                    door_edges.append(((row - 1, c), (row, c)))
                else:
                    col, r = a, b
                    door_edges.append(((r, col - 1), (r, col)))

        # ---- room interiors + anchors ----
        room_interiors = []
        room_anchors = []
        for (r0, c0, h, w) in rooms:
            interior = set()
            for r in range(r0 + 1, r0 + h - 1):
                for c in range(c0 + 1, c0 + w - 1):
                    interior.add((r, c))
            if not interior:
                return None
            room_interiors.append(interior)
            room_anchors.append(local_rng.choice(list(interior)))

    
        # ---- agent + goal ----
        if not skip_room_phase:
            if num_rooms >= 2:
                ai = local_rng.randrange(num_rooms)
                gi = (ai + local_rng.randint(1, num_rooms - 1)) % num_rooms
                agent = room_anchors[ai]
                goal = room_anchors[gi]
            elif num_rooms == 1:
                interior_list = list(room_interiors[0])
                if len(interior_list) < 2:
                    return None
                local_rng.shuffle(interior_list)
                agent, goal = interior_list[0], interior_list[1]
            else:
                all_cells = [(r, c) for r in range(N_) for c in range(N_)]
                agent = local_rng.choice(all_cells)
                goal = local_rng.choice([p for p in all_cells if p != agent])
        # In easy mode, agent and goal were already set above (skip_room_phase=True)


        # ---- protected (doors + neighbors, agent/goal + neighbors) ----
        protected = set()

        def protect_cell(rc):
            r, c = rc
            protected.add((r, c))
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < N_ and 0 <= cc < N_:
                    protected.add((rr, cc))

        for a, b in door_edges:
            if 0 <= a[0] < N_ and 0 <= a[1] < N_:
                protect_cell(a)
            if 0 <= b[0] < N_ and 0 <= b[1] < N_:
                protect_cell(b)
        protect_cell(agent)
        protect_cell(goal)

        # ---- compute wall-adjacent from ROOM walls only (for workstation spacing) ----
        wall_adj_room = set()
        for r in range(N_):
            for c in range(N_):
                if (
                    vwall_room[c, r]
                    or vwall_room[c + 1, r]
                    or hwall_room[r, c]
                    or hwall_room[r + 1, c]
                ):
                    wall_adj_room.add((r, c))

        # ---- workstation placement ----
        all_cells_set = {(r, c) for r in range(N_) for c in range(N_)}
        all_room_cells = set()
        for (r0, c0, h, w) in rooms:
            for rr in range(r0, r0 + h):
                for cc in range(c0, c0 + w):
                    all_room_cells.add((rr, cc))

        if add_workstations:
            CLEAR = 2

            def expand_manhattan(cells, dist):
                out = set(cells)
                for _ in range(dist):
                    new = set(out)
                    for (r, c) in out:
                        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < N_ and 0 <= cc < N_:
                                new.add((rr, cc))
                    out = new
                return out

            near_room_walls = expand_manhattan(wall_adj_room, CLEAR)

            boundary_band = set()
            for r in range(N_):
                for c in range(N_):
                    if r < CLEAR or c < CLEAR or r >= N_ - CLEAR or c >= N_ - CLEAR:
                        boundary_band.add((r, c))

            forbidden_centers = near_room_walls | boundary_band | all_room_cells | protected
            candidate_centers = list(all_cells_set - forbidden_centers)
            local_rng.shuffle(candidate_centers)

            max_possible = max(1, (N_ * N_) // 55)
            desired_ws = max(1, min(max_possible, (num_rooms + num_obstacles) // 2))
            workstations = []

            def ws_walls_conflict(changes):
                for typ, idx in changes:
                    if typ == "v":
                        col, rr = idx
                        if vwall_room[col, rr] or vwall_ws[col, rr]:
                            return True
                    else:
                        row, cc = idx
                        if hwall_room[row, cc] or hwall_ws[row, cc]:
                            return True
                return False

            def apply_ws(changes):
                for typ, idx in changes:
                    if typ == "v":
                        vwall_ws[idx] = True
                    else:
                        hwall_ws[idx] = True

            def undo_ws(changes):
                for typ, idx in changes:
                    if typ == "v":
                        vwall_ws[idx] = False
                    else:
                        hwall_ws[idx] = False

            def try_place_workstation(center_rc, arm_len):
                r, c = center_rc
                if not (2 <= r <= N_ - 3 and 2 <= c <= N_ - 3):
                    return None

                missing = local_rng.choice(["up", "down", "left", "right"])
                col_line = c + 1
                row_line = r + 1

                footprint = {(r, c)}
                for k in range(1, arm_len + 1):
                    if missing != "up":
                        footprint.add((r - k, c))
                    if missing != "down":
                        footprint.add((r + k, c))
                    if missing != "left":
                        footprint.add((r, c - k))
                    if missing != "right":
                        footprint.add((r, c + k))

                if any((rr, cc) in forbidden_centers for (rr, cc) in footprint):
                    return None
                if any(not (0 <= rr < N_ and 0 <= cc < N_) for (rr, cc) in footprint):
                    return None

                changes = []
                if missing != "up":
                    for rr in range(max(1, r - arm_len), r):
                        changes.append(("v", (col_line, rr)))
                if missing != "down":
                    for rr in range(r, min(N_ - 1, r + arm_len)):
                        changes.append(("v", (col_line, rr)))

                if missing != "left":
                    for cc in range(max(1, c - arm_len), c):
                        changes.append(("h", (row_line, cc)))
                if missing != "right":
                    for cc in range(c, min(N_ - 1, c + arm_len)):
                        changes.append(("h", (row_line, cc)))

                if ws_walls_conflict(changes):
                    return None

                vtot_before = vwall_total()
                htot_before = hwall_total()

                def endpoint_touches_existing():
                    if missing != "up":
                        rr0 = max(1, r - arm_len)
                        if rr0 - 1 >= 0 and vtot_before[col_line, rr0 - 1]:
                            return True
                    if missing != "down":
                        rr1 = min(N_ - 2, r + arm_len - 1)
                        if rr1 + 1 <= N_ - 1 and vtot_before[col_line, rr1 + 1]:
                            return True
                    if missing != "left":
                        cc0 = max(1, c - arm_len)
                        if cc0 - 1 >= 0 and htot_before[row_line, cc0 - 1]:
                            return True
                    if missing != "right":
                        cc1 = min(N_ - 2, c + arm_len - 1)
                        if cc1 + 1 <= N_ - 1 and htot_before[row_line, cc1 + 1]:
                            return True
                    return False

                if endpoint_touches_existing():
                    return None

                apply_ws(changes)

                reach = reachable_set(N_, agent, blocked, vwall_total(), hwall_total())
                ok = (goal in reach) and all(a in reach for a in room_anchors)

                if not ok:
                    undo_ws(changes)
                    return None

                extra_protect = set()
                for (rr, cc) in footprint:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            rrr, ccc = rr + dr, cc + dc
                            if 0 <= rrr < N_ and 0 <= ccc < N_:
                                extra_protect.add((rrr, ccc))

                return {
                    "center": (r, c),
                    "arm": int(arm_len),
                    "extra_protect": extra_protect,
                    "missing": missing,
                }

            attempts = 0
            for rc in candidate_centers:
                if len(workstations) >= desired_ws:
                    break
                attempts += 1
                if attempts > 5000:
                    break

                arm = 1 if N_ < 14 else 2
                if N_ >= 20 and local_rng.random() < 0.25:
                    arm = 3

                placed = try_place_workstation(rc, arm)
                if placed is None:
                    continue

                protected |= placed["extra_protect"]
                workstations.append(
                    {"id": len(workstations) + 1, "center": placed["center"], "arm": placed["arm"]}
                )

            if len(workstations) < 1:
                return None
        else:
            workstations = []

        # ---- wall-adjacent cells from TOTAL walls (rooms + workstations) for obstacle rules ----
        vtot = vwall_total()
        htot = hwall_total()

        wall_adj_total = set()
        for r in range(N_):
            for c in range(N_):
                if (
                    vtot[c, r]
                    or vtot[c + 1, r]
                    or htot[r, c]
                    or htot[r + 1, c]
                ):
                    wall_adj_total.add((r, c))

        # ---- bins for obstacle distribution ----
        room_bins = [set(interior) - wall_adj_total for interior in room_interiors]
        all_room_interior = set().union(*room_interiors) if room_interiors else set()
        outside_bin = (all_cells_set - all_room_interior) - wall_adj_total
        bins = room_bins + [outside_bin]
        bin_count = len(bins)

        base = num_obstacles // bin_count
        rem = num_obstacles % bin_count
        targets = [base + (1 if i < rem else 0) for i in range(bin_count)]

        order = []
        for i, t in enumerate(targets):
            order += [i] * t
        local_rng.shuffle(order)

        def all_valid_rects(allowed_cells, h_range, w_range):
            rects = []
            if not allowed_cells:
                return rects
            for hh in range(h_range[0], h_range[1] + 1):
                for ww in range(w_range[0], w_range[1] + 1):
                    if hh <= 0 or ww <= 0 or hh > N_ or ww > N_:
                        continue
                    for r0 in range(0, N_ - hh + 1):
                        for c0 in range(0, N_ - ww + 1):
                            cells = [(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)]
                            if any((r, c) not in allowed_cells for (r, c) in cells):
                                continue
                            if any((r, c) in protected for (r, c) in cells):
                                continue
                            if any(blocked[r, c] for (r, c) in cells):
                                continue
                            rects.append((r0, c0, hh, ww, cells))
            return rects

        def try_place_in_allowed(allowed_cells, h_range, w_range):
            rects = all_valid_rects(allowed_cells, h_range, w_range)
            local_rng.shuffle(rects)
            for (r0, c0, hh, ww, cells) in rects:
                for (r, c) in cells:
                    blocked[r, c] = True
                reach = reachable_set(N_, agent, blocked, vtot, htot)
                ok = (goal in reach) and all(a in reach for a in room_anchors)
                if ok:
                    return (r0, c0, hh, ww)
                for (r, c) in cells:
                    blocked[r, c] = False
            return None

        # ---- place obstacles ----
        obstacles = []
        global_allowed = all_cells_set - wall_adj_total

        for assigned_bin in order:
            bin_try = [assigned_bin] + [i for i in range(bin_count) if i != assigned_bin]
            local_rng.shuffle(bin_try[1:])

            placed = None
            for i in bin_try:
                allowed = bins[i]
                if i < len(room_bins):
                    rs = [p[0] for p in allowed] if allowed else []
                    cs = [p[1] for p in allowed] if allowed else []
                    if not rs or not cs:
                        continue
                    span_h = max(rs) - min(rs) + 1
                    span_w = max(cs) - min(cs) + 1
                    h_range = (1, max(1, min(2, span_h)))
                    w_range = (1, max(1, min(3, span_w)))
                else:
                    h_range = obstacle_h_range
                    w_range = obstacle_w_range

                placed = try_place_in_allowed(allowed, h_range, w_range)
                if placed is not None:
                    obstacles.append(placed)
                    break

            if placed is None:
                placed = try_place_in_allowed(global_allowed, obstacle_h_range, obstacle_w_range)
                if placed is None:
                    return None
                obstacles.append(placed)
        # ---- compute BFS shortest path waypoints (agent -> goal) ----
        bfs_path_rc = bfs_shortest_path(N_, agent, goal, blocked, vtot, htot)
        if not bfs_path_rc:
            # should not happen because we enforce reachability, but safe fallback
            return None

        # ---------------- render image ----------------
        margin_left = 55
        margin_top = 25
        margin_bottom = 55
        margin_right = 20

        IMG_W = margin_left + N_ * cell_px + margin_right
        IMG_H = margin_top + N_ * cell_px + margin_bottom

        def gx(c):
            return margin_left + c * cell_px

        def gy(r):
            return margin_top + r * cell_px

        def make_canvas():
            img = Image.new("RGB", (IMG_W, IMG_H), "white")
            d = ImageDraw.Draw(img)
            return img, d

        def draw_grid(d):
            for i in range(N_ + 1):
                d.line([(gx(0), gy(i)), (gx(N_), gy(i))], fill=GRID, width=1)
                d.line([(gx(i), gy(0)), (gx(i), gy(N_))], fill=GRID, width=1)

        def draw_axes(d):
            y = margin_top + N_ * cell_px + 10

            # X axis (0-indexed)
            for xline in range(0, N_):
                x = gx(xline)
                d.text((x - 4, y), str(xline), fill=(80, 80, 80), font=font_small)

            # Y axis (1-indexed, top-down corrected)
            x = 10
            for yline in range(0, N_):
                y_pos = gy(N_ - 1 - yline)
                d.text((x, y_pos - 7), str(yline + 1), fill=(80, 80, 80), font=font_small)

            d.text(
                (margin_left + (N_ * cell_px) // 2 - 10, margin_top + N_ * cell_px + 30),
                "X",
                fill=(80, 80, 80),
                font=font,
            )
            d.text(
                (25, margin_top + (N_ * cell_px) // 2 - 10),
                "Y",
                fill=(80, 80, 80),
                font=font,
            )


        def paint_vwall(d, col, r0, r1, px):
            x = gx(col)
            y0 = gy(r0)
            y1 = gy(r1)
            half = px // 2
            d.rectangle([x - half, y0 - half, x + half, y1 + half], fill=BLACK)

        def paint_hwall(d, row, c0, c1, px):
            y = gy(row)
            x0 = gx(c0)
            x1 = gx(c1)
            half = px // 2
            d.rectangle([x0 - half, y - half, x1 + half, y + half], fill=BLACK)

        def render_walls(d, vwall, hwall, px):
            for col in range(N_ + 1):
                for r in range(N_):
                    if vwall[col, r]:
                        paint_vwall(d, col, r, r + 1, px)
            for row in range(N_ + 1):
                for c in range(N_):
                    if hwall[row, c]:
                        paint_hwall(d, row, c, c + 1, px)

        def fill_obstacle_cells(d, r0, c0, h, w, inset=1):
            colour_obstacle = (128, 128, 128)
            x0 = gx(c0) + inset
            y0 = gy(r0) + inset
            x1 = gx(c0 + w) - inset
            y1 = gy(r0 + h) - inset
            d.rectangle([x0, y0, x1, y1], fill=colour_obstacle)

        def circle_in_cell(d, r, c, color):
            cx = gx(c)
            cy = gy(r)
            rad = int(cell_px * 0.15)
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=color)


        img, d = make_canvas()
        draw_grid(d)

        # draw room walls thick
        render_walls(d, vwall_room, hwall_room, wall_px)

        # draw workstation walls thinner
        ws_wall_px = max(2, wall_px // 3)
        render_walls(d, vwall_ws, hwall_ws, ws_wall_px)

        for (r0, c0, hh, ww) in obstacles:
            fill_obstacle_cells(d, r0, c0, hh, ww)

        for ws in workstations:
            r, c = ws["center"]
            label = f"WORK STATION {ws['id']}"
            tx = gx(c) + int(cell_px * 0.10)
            ty = gy(r) - int(cell_px * 0.35)
            d.text((tx, ty), label, fill=(20, 20, 20), font=font_small)

        circle_in_cell(d, agent[0], agent[1], (40, 80, 255))
        circle_in_cell(d, goal[0], goal[1], (50, 180, 90))

        draw_axes(d)

        def to_xy(rc):
            r, c = rc
            return {"x": int(c + 1), "y": int(N_ - r), "r": int(r), "c": int(c)}

        def to_xy_center(rc):
            r, c = rc
            # center of the cell (not on grid lines)
            return {
                "x": float(c + 0.5),
                "y": float((N_ - r) - 0.5),
                "r": int(r),
                "c": int(c),
            }


        meta = {
            "N": int(N_),
            "seed": int(local_seed),
            "num_rooms": int(num_rooms),
            "num_obstacles": int(num_obstacles),
            "num_workstations": int(len(workstations)),
            "agent": to_xy(agent),
            "goal": to_xy(goal),
            "rooms": [{"r0": int(r0), "c0": int(c0), "h": int(h), "w": int(w)} for (r0, c0, h, w) in rooms],
            "doors": [{"kind": k, "a": int(a), "b": int(b)} for (k, a, b) in room_doors],
            "obstacles": [{"r0": int(r0), "c0": int(c0), "h": int(h), "w": int(w)} for (r0, c0, h, w) in obstacles],
            "workstations": [{"id": int(ws["id"]), "center": to_xy(ws["center"]), "arm": int(ws["arm"])} for ws in workstations],
            "bfs_path_len": int(len(bfs_path_rc)),
            "bfs_path": [to_xy_center(rc) for rc in bfs_path_rc],
            "vwall_room": vwall_room.astype(int).tolist(),
            "hwall_room": hwall_room.astype(int).tolist(),

        }

        return img, meta

    # ---------- auto-grow outer loop ----------
    N_current = int(N)
    while True:
        for _ in range(max_global_restarts):
            local_seed = rng.randrange(10**9)
            res = build_once(N_current, local_seed)
            if res is None:
                continue

            img, meta = res

            # 🔴 CHANGE IS HERE
            filename_seed = seed if seed is not None else local_seed

            out_png = os.path.join(
                out_dir,
                f"env_{stamp()}_seed{filename_seed}_N{N_current}_R{num_rooms}_O{num_obstacles}.png"
            )
            out_json = out_png.replace(".png", ".json")

            img.save(out_png)
            with open(out_json, "w") as f:
                json.dump(meta, f, indent=2)

            print("Saved:", out_png)
            return out_png, out_json

        if not auto_grow or N_current >= max_N:
            raise RuntimeError("Could not generate a valid environment.")
        N_current = min(max_N, N_current + grow_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=3)
    ap.add_argument("--obstacles", type=int, default=4)
    ap.add_argument("--workstations", type=str, default="n", help="y to add workstations, n to disable")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--outdir", type=str, default="logs/warehouse")
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard"], help="Environment difficulty level: easy, medium, or hard (default: hard)")

    args = ap.parse_args()
    use_ws = args.workstations.strip().lower() in ["y", "yes", "1", "true"]
    difficulty = args.difficulty.strip().lower()


    if args.grid >= 16:
        cell_px, wall_px = 40, 8
        restarts = 1200
        oh = (1, 4)
        ow = (1, 6)
    else:
        cell_px, wall_px = 70, 12
        restarts = 1600
        oh = (1, 2)
        ow = (1, 4)

    generate_env(
        N=args.grid,
        num_rooms=args.rooms,
        num_obstacles=args.obstacles,
        seed=args.seed,
        out_dir=args.outdir,
        cell_px=cell_px,
        wall_px=wall_px,
        obstacle_h_range=oh,
        obstacle_w_range=ow,
        max_global_restarts=restarts,
        auto_grow=True,
        grow_step=3,
        max_N=60,
        add_workstations=use_ws,
        difficulty=difficulty, 
    )


if __name__ == "__main__":
    main()