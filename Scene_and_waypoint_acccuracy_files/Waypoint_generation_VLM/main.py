# main.py
# Process images; for each image, run 4 models in parallel:
#   AWS: haiku, sonnet
#   OpenAI: gpt-4o, gpt-4o-mini
# Writes:
#   new_minor_grid_off/results_aws_haiku.xlsx
#   new_minor_grid_off/results_aws_sonnet.xlsx
#   new_minor_grid_off/results_gpt_gpt-4o.xlsx
#   new_minor_grid_off/results_gpt_gpt-4o-mini.xlsx
# Also saves per-model overlays under:
#   IMG_DIR/overlays_haiku/, overlays_sonnet/, overlays_gpt-4o/, overlays_gpt-4o-mini/
#
# pip install pandas openpyxl matplotlib pillow

import matplotlib
matplotlib.use("Agg")  # non-GUI backend (safe for threads)

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import pandas as pd
import time
import math

from extract_scene_aws import extract_scene_from_image as extract_aws
from extract_scene_gpt import extract_scene_from_image as extract_gpt

# ---- plotting (inline helper, no extra file) ----
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

IMG_DIR = Path("new_minor_grid_off")

MODELS: List[Tuple[str, str]] = [
    ("aws", "haiku"),
    ("aws", "sonnet"),
    ("gpt", "gpt-4o"),
    ("gpt", "gpt-4o-mini"),
]

# ---------- small formatters ----------
def _fmt_xy(xy): return f"({float(xy[0]):.2f},{float(xy[1]):.2f})"
def _fmt_xy_list(lst): return "[" + ", ".join(_fmt_xy(p) for p in lst) + "]"
def _fmt_vertices(verts): return "[" + ", ".join(_fmt_xy(v) for v in verts) + "]"
def _fmt_seg_pairs(pairs):  # [([x1,y1],[x2,y2]), ...]
    return "[" + ", ".join(f"{_fmt_xy(a)}-{_fmt_xy(b)}" for (a,b) in pairs) + "]"

def _write_excel(rows: List[Dict], out_path: Path):
    max_obs = 0
    for r in rows:
        for k in r.keys():
            if k.startswith("Obstacle"):
                idx = int(k.replace("Obstacle", ""))
                max_obs = max(max_obs, idx)
    cols = ["Image", "Model", "Agent", "Goal", "Waypoints",
            "Invalid points", "Segment intersect"] + [f"Obstacle{i}" for i in range(1, max_obs + 1)]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)

# -------- geometry helpers --------
def _point_on_edge(pt, poly):
    """True if pt lies on any edge of polygon 'poly' (including vertices)."""
    x, y = pt
    n = len(poly)
    def on_seg(p,q,r):
        return (min(p[0],r[0]) - 1e-9 <= q[0] <= max(p[0],r[0]) + 1e-9 and
                min(p[1],r[1]) - 1e-9 <= q[1] <= max(p[1],r[1]) + 1e-9 and
                abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])) < 1e-9)
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        if on_seg(a, (x,y), b):
            return True
    return False

def _point_in_poly_strict(pt, poly):
    """Strict interior test (excludes boundary)."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1):
            inside = not inside
    return inside

def _seg_intersect(a,b,c,d):
    """Proper segment intersection (incl. colinear-on-segment)."""
    def orient(p,q,r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def on_seg(p,q,r):
        return (min(p[0],r[0]) - 1e-9 <= q[0] <= max(p[0],r[0]) + 1e-9 and
                min(p[1],r[1]) - 1e-9 <= q[1] <= max(p[1],r[1]) + 1e-9)
    o1 = orient(a,b,c); o2 = orient(a,b,d); o3 = orient(c,d,a); o4 = orient(c,d,b)
    if (o1*o2 < 0) and (o3*o4 < 0): return True
    if abs(o1) < 1e-9 and on_seg(a,c,b): return True
    if abs(o2) < 1e-9 and on_seg(a,d,b): return True
    if abs(o3) < 1e-9 and on_seg(c,a,d): return True
    if abs(o4) < 1e-9 and on_seg(c,b,d): return True
    return False

def _poly_edges(poly):
    return [(poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly))]

# ---------- violation calculators (agent/goal endpoint exceptions) ----------
def _invalid_points_and_segments(scene: Dict):
    """
    Return (invalid_points, bad_segments) with endpoint exceptions:
      - Goal ON an obstacle edge is allowed (not an invalid point).
      - Final hop may 'touch' an obstacle only at the GOAL endpoint (no grazing).
      - Agent ON an obstacle edge is allowed (not an invalid point).
      - First hop may 'touch' an obstacle only at the AGENT endpoint (no grazing).
    Grazing (running along an obstacle edge for >0 length) is NOT allowed.
    """
    obstacles = scene.get("obstacles", [])
    wps = scene.get("waypoints", []) or []
    goal = tuple(scene["goal"])
    agent = tuple(scene["agent"])

    # tiny helper: is the segment grazing along the edge right after endpoint ep?
    def _grazes_from_endpoint(ep, other, poly):
        ex, ey = ep; ox, oy = other
        dx, dy = (ox - ex, oy - ey)
        norm = math.hypot(dx, dy) or 1.0
        t = 1e-3 / norm
        probe = (ex + dx * t, ey + dy * t)
        return _point_on_edge(probe, poly)

    # invalid points (strictly inside, or on-edge) BUT allow goal/agent on-edge
    def point_invalid(p):
        pt = tuple(p)
        for ob in obstacles:
            poly = ob["vertices"]
            if _point_in_poly_strict(pt, poly):
                return True
            if _point_on_edge(pt, poly):
                if pt == goal or pt == agent:
                    return False  # allow endpoints on boundary
                return True
        return False

    invalid_points = [p for p in wps if point_invalid(p)]

    # bad segments with endpoint-only allowances at agent/goal (no grazing beyond endpoint)
    def seg_hits_obstacles_with_endpoint_allowance(p, q):
        p = tuple(p); q = tuple(q)
        for ob in obstacles:
            poly = ob["vertices"]
            # any intersection?
            hit = False
            for e0, e1 in _poly_edges(poly):
                if _seg_intersect(p, q, e0, e1):
                    hit = True
                    break
            if not hit:
                continue

            # allow final hop touching at goal endpoint only
            if q == goal and _point_on_edge(q, poly):
                if not _point_in_poly_strict(p, poly):
                    if not _grazes_from_endpoint(q, p, poly):
                        continue  # allowed

            # allow first hop touching at agent endpoint only
            if p == agent and _point_on_edge(p, poly):
                if not _point_in_poly_strict(q, poly):
                    if not _grazes_from_endpoint(p, q, poly):
                        continue  # allowed

            # otherwise, it's a bad segment
            return True

        return False

    bad_segments = []
    for i in range(len(wps) - 1):
        p, q = wps[i], wps[i+1]
        if seg_hits_obstacles_with_endpoint_allowance(p, q):
            bad_segments.append((p, q))

    return invalid_points, bad_segments

# ---------- overlay drawing ----------
def _save_scene_png(scene: Dict, out_path: Path):
    """Render a clean scene view from extracted JSON (not the original PNG)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    # world bounds
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor("white")
    ax.grid(True, which='both', linewidth=0.5, alpha=0.3)

    # obstacles
    obstacles = scene.get("obstacles", [])
    for ob in obstacles:
        verts = ob.get("vertices", [])
        if len(verts) >= 3:
            poly = MplPolygon(verts, closed=True, fill=False, linewidth=2, edgecolor="black")
            ax.add_patch(poly)

    # agent & goal
    ax.scatter([scene["agent"][0]],[scene["agent"][1]], s=40, label="agent", color="tab:blue")
    ax.scatter([scene["goal"][0]],[scene["goal"][1]], s=60, marker="*", label="goal", color="tab:orange")

    # violations (reuse same logic as Excel)
    invalid_points, bad_segments = _invalid_points_and_segments(scene)

    # waypoints
    wps = scene.get("waypoints", []) or []
    if len(wps) >= 1:
        bad_set = set((tuple(p), tuple(q)) for (p,q) in bad_segments)

        # draw segments (red if intersect, else blue)
        for i in range(len(wps) - 1):
            p, q = wps[i], wps[i+1]
            is_bad = (tuple(p), tuple(q)) in bad_set
            ax.plot([p[0], q[0]], [p[1], q[1]],
                    linewidth=2,
                    color=("red" if is_bad else "tab:blue"),
                    zorder=2)

        # draw points (red if invalid)
        bad_pts = {tuple(p) for p in invalid_points}
        good_x, good_y, bad_x, bad_y = [], [], [], []
        for p in wps:
            if tuple(p) in bad_pts:
                bad_x.append(p[0]); bad_y.append(p[1])
            else:
                good_x.append(p[0]); good_y.append(p[1])

        if good_x:
            ax.scatter(good_x, good_y, s=14, color="tab:blue", zorder=3)
        if bad_x:
            ax.scatter(bad_x, bad_y, s=16, color="red", zorder=4, label="invalid wp/segment")

    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# ---------- one model call ----------
def _call_one(model_family: str, model_name: str, img: Path) -> Tuple[str, Dict]:
    caller = extract_aws if model_family == "aws" else extract_gpt
    scene = caller(str(img), model=model_name)

    # Compute violations for Excel
    invalid_points, bad_segments = _invalid_points_and_segments(scene)

    # Row for Excel
    row = {
        "Image": img.name,
        "Model": model_name,
        "Agent": _fmt_xy(scene["agent"]),
        "Goal": _fmt_xy(scene["goal"]),
        "Waypoints": _fmt_xy_list(scene.get("waypoints", [])),
        "Invalid points": _fmt_xy_list(invalid_points),
        "Segment intersect": _fmt_seg_pairs(bad_segments),
    }
    for i, ob in enumerate(scene.get("obstacles", []), start=1):
        row[f"Obstacle{i}"] = _fmt_vertices(ob["vertices"])

    # Save overlay PNG (rendered from scene JSON)
    outdir = IMG_DIR / f"overlays_{model_name}"
    out_png = outdir / (img.stem + ".png")
    try:
        _save_scene_png(scene, out_png)
    except Exception as e:
        # Non-fatal: record a note in the row if saving fails
        row["Segment intersect"] += f"  (overlay_err: {e})"

    return model_name, row

# ---------- main ----------
def main():
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {IMG_DIR.resolve()}")
    images = sorted(IMG_DIR.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images in {IMG_DIR.resolve()}")

    per_model_rows: Dict[str, List[Dict]] = {name: [] for _, name in MODELS}

    for img in images:
        print(f"\n=== {img.name} ===")
        with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
            futures = {pool.submit(_call_one, fam, name, img): (fam, name) for (fam, name) in MODELS}
            for fut in as_completed(futures):
                fam, name = futures[fut]
                try:
                    _, row = fut.result()
                    per_model_rows[name].append(row)
                    print(f"[OK] {name}")
                except Exception as e:
                    per_model_rows[name].append(
                        {
                            "Image": img.name, "Model": name,
                            "Agent": "ERR", "Goal": f"ERR: {e}",
                            "Waypoints": "", "Invalid points": "", "Segment intersect": ""
                        }
                    )
                    print(f"[ERR] {name} -> {e}")

    _write_excel(per_model_rows["haiku"],       IMG_DIR / "results_aws_haiku.xlsx")
    _write_excel(per_model_rows["sonnet"],      IMG_DIR / "results_aws_sonnet.xlsx")
    _write_excel(per_model_rows["gpt-4o"],      IMG_DIR / "results_gpt_gpt-4o.xlsx")
    _write_excel(per_model_rows["gpt-4o-mini"], IMG_DIR / "results_gpt_gpt-4o-mini.xlsx")
    print("\n[OK] Wrote 4 Excel files to:", IMG_DIR.resolve())

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed = (time.perf_counter() - start_time)/60
    print(f"\nTotal execution time: {elapsed:.2f} minutes")
