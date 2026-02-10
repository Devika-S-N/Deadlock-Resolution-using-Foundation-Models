#!/usr/bin/env python3
import ast
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# ----------------------- CONFIG -----------------------
MASTER_FILE = "environments_log.xlsx"
# Child files (exact names you gave)
CHILD_FILES = [
    "results_aws_haiku.xlsx",
    "results_aws_sonnet.xlsx",
    "results_gpt_gpt-4o.xlsx",
    "results_gpt_gpt-4o-mini.xlsx",
    "results_gpt_gpt-5.xlsx",
]
# Tolerance for coordinate comparison (applies to Agent, Goal, and obstacle vertices)
TOL = 0.5  # adjust if needed

# Expected master columns (case-insensitive match; we'll resolve actual names)
MASTER_AGENT_COL = "agent  position"   # note the double space as provided
MASTER_GOAL_COL  = "goal position"
MASTER_OBS_COLS  = [f"obstacle{i}" for i in range(1, 7)]

# Expected child columns (case-insensitive match)
CHILD_IMAGE_COL = "image"
CHILD_MODEL_COL = "model"
CHILD_AGENT_COL = "agent"
CHILD_GOAL_COL  = "goal"
CHILD_OBS_COLS  = [f"obstacle{i}" for i in range(1, 7)]

# Output Excel
OUTPUT_FILE = "results_accuracy.xlsx"
# ------------------------------------------------------


# ---------- Parsing helpers ----------
def _safe_literal_eval(s: Any):
    """Safely parse lists/tuples from string. Returns None for NaN/empty."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    if isinstance(s, (list, tuple, dict)):
        return s
    if isinstance(s, str):
        txt = s.strip()
        if txt == "" or txt.lower() in {"none", "nan"}:
            return None
        # Normalize some common notations
        # e.g., "(x,y)" vs "[(x,y)]"; we let ast handle both list/tuple content
        try:
            return ast.literal_eval(txt)
        except Exception:
            # If it's like "(x, y)" without brackets, try to wrap
            if txt.startswith("(") and txt.endswith(")"):
                try:
                    return ast.literal_eval(f"[{txt}]")
                except Exception:
                    return None
            return None
    return None


def _to_point_list(value: Any) -> List[Tuple[float, float]]:
    """
    Convert a variety of user notations to a list of (x,y) floats.
    Accepts:
      - "(x,y)"
      - "[(x,y),(a,b)]"
      - [(x,y), (a,b)]
      - [[x,y], [a,b]]
    Returns [] for None/unparseable/empty.
    """
    parsed = _safe_literal_eval(value)
    if parsed is None:
        return []

    def _as_xy(p):
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                return (float(p[0]), float(p[1]))
            except Exception:
                return None
        return None

    pts: List[Tuple[float, float]] = []
    if isinstance(parsed, (list, tuple)):
        # Could be a list of pairs or a single pair wrapped as list (from "(x,y)")
        if len(parsed) == 2 and all(isinstance(c, (int, float)) for c in parsed):
            xy = _as_xy(parsed)
            if xy:
                pts.append(xy)
        else:
            for item in parsed:
                xy = _as_xy(item)
                if xy:
                    pts.append(xy)
    else:
        # Not a list/tuple -> not supported
        pass
    return pts


def _to_single_point(value: Any) -> Optional[Tuple[float, float]]:
    """
    Parse Agent/Goal which might be "(x,y)" or "[(x,y)]" or [x,y].
    Returns (x,y) or None.
    """
    pts = _to_point_list(value)
    if len(pts) == 1:
        return pts[0]
    # If more than one or empty, treat as None
    return None


# ---------- Comparison helpers ----------
def _within_tol(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol

def _point_match(p: Tuple[float, float], q: Tuple[float, float], tol: float) -> bool:
    return _within_tol(p[0], q[0], tol) and _within_tol(p[1], q[1], tol)

def _point_list_match_unordered(A: List[Tuple[float, float]],
                                B: List[Tuple[float, float]],
                                tol: float) -> bool:
    """
    Check if two lists of points match ignoring order, with per-coordinate tolerance.
    Greedy matching is used (no external dependencies).
    """
    if len(A) != len(B):
        return False
    if not A and not B:
        return True

    used = [False] * len(B)
    for p in A:
        found = False
        # Match to nearest within tol (greedy)
        best_j = -1
        best_dist = float("inf")
        for j, q in enumerate(B):
            if used[j]:
                continue
            # quick reject if outside per-coordinate tol
            if not _point_match(p, q, tol):
                continue
            d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            found = True
        if not found:
            return False
    return True

def compare_agent_goal(master_val, child_val, tol: float) -> bool:
    mp = _to_single_point(master_val)
    cp = _to_single_point(child_val)
    if mp is None or cp is None:
        return False
    return _point_match(mp, cp, tol)

def compare_obstacle(master_val, child_val, tol: float) -> bool:
    """
    Compare one obstacle (list of vertex points) with tolerance and order-insensitive.
    """
    mpts = _to_point_list(master_val)
    cpts = _to_point_list(child_val)
    return _point_list_match_unordered(mpts, cpts, tol)


# ---------- Column resolution ----------
def resolve_columns(df: pd.DataFrame, targets: List[str]) -> Dict[str, str]:
    """
    Map desired logical column names (targets) to actual columns in df (case-insensitive, ignores spaces differences).
    Returns dict logical_name -> actual_column_name. Missing remain unmapped.
    """
    norm_map = {}
    def norm(s: str) -> str:
        return "".join(str(s).strip().lower().split())

    actuals = {norm(c): c for c in df.columns}
    for t in targets:
        key = norm(t)
        if key in actuals:
            norm_map[t] = actuals[key]
    return norm_map


# ---------- Scoring ----------
def score_child_against_master(master_df: pd.DataFrame, child_df: pd.DataFrame, tol: float) -> (pd.DataFrame, float):
    """
    Align rows by image index (row number) and compute per-row correctness:
      - Agent: 1 if match else 0
      - Goal: 1 if match else 0
      - Obstacles: count of matches using one-to-one mapping between master and child obstacles (any index)
    Total items per row = 2 + (#non-empty master obstacles).
    Returns:
      details_df: per-row breakdown
      overall_accuracy: sum(correct)/sum(total)
    """
    # Resolve master columns
    m_cols = resolve_columns(master_df, [MASTER_AGENT_COL, MASTER_GOAL_COL] + MASTER_OBS_COLS)
    # Resolve child columns
    c_cols = resolve_columns(child_df, [CHILD_IMAGE_COL, CHILD_MODEL_COL, CHILD_AGENT_COL, CHILD_GOAL_COL] + CHILD_OBS_COLS)

    # Fallback if some names slightly differ; raise if core ones missing.
    m_agent = m_cols.get(MASTER_AGENT_COL)
    m_goal  = m_cols.get(MASTER_GOAL_COL)
    if not m_agent or not m_goal:
        raise ValueError("Could not find master Agent/Goal columns. Found columns: " + ", ".join(master_df.columns))

    c_agent = c_cols.get(CHILD_AGENT_COL)
    c_goal  = c_cols.get(CHILD_GOAL_COL)
    if not c_agent or not c_goal:
        raise ValueError("Could not find child Agent/Goal columns. Found columns: " + ", ".join(child_df.columns))

    m_obs_cols = [m_cols.get(col) for col in MASTER_OBS_COLS if m_cols.get(col)]
    c_obs_cols = [c_cols.get(col) for col in CHILD_OBS_COLS if c_cols.get(col)]

    # Prepare results
    rows = []
    total_correct_all = 0
    total_items_all = 0

    n = min(len(master_df), len(child_df))
    for idx in range(n):
        m_row = master_df.iloc[idx]
        c_row = child_df.iloc[idx]

        # Per row tracking
        correct = 0
        total_items = 0

        # Agent
        agent_ok = compare_agent_goal(m_row[m_agent], c_row[c_agent], tol)
        correct += 1 if agent_ok else 0
        total_items += 1

        # Goal
        goal_ok = compare_agent_goal(m_row[m_goal], c_row[c_goal], tol)
        correct += 1 if goal_ok else 0
        total_items += 1

        # Obstacles: build a bipartite-like greedy one-to-one mapping
        # First list of master obstacles that are non-empty
        master_obs_vals = []
        for col in m_obs_cols:
            pts = _to_point_list(m_row[col]) if col in m_row else []
            if len(pts) > 0:  # consider only non-empty obstacles
                master_obs_vals.append(m_row[col])

        # Child obstacle candidates
        child_obs_vals = []
        for col in c_obs_cols:
            pts = _to_point_list(c_row[col]) if col in c_row else []
            # We allow empty child obstacles; they simply won't match.
            child_obs_vals.append(c_row[col] if len(pts) > 0 else None)

        # Count obstacle items only for non-empty master obstacles
        total_items += len(master_obs_vals)
        # Create availability mask for child obs
        child_used = [False] * len(child_obs_vals)
        obs_matches = 0

        for m_val in master_obs_vals:
            matched = False
            # Try to match with any unused child obstacle
            for j, c_val in enumerate(child_obs_vals):
                if child_used[j] or c_val is None:
                    continue
                if compare_obstacle(m_val, c_val, tol):
                    child_used[j] = True
                    matched = True
                    break
            if matched:
                obs_matches += 1

        correct += obs_matches

        # Track totals
        total_correct_all += correct
        total_items_all += total_items

        rows.append({
            "Row": idx+1,
            "Agent_Correct": int(agent_ok),
            "Goal_Correct": int(goal_ok),
            "Master_Obstacle_Count": len(master_obs_vals),
            "Matched_Obstacles": obs_matches,
            "Row_Accuracy": (correct / total_items) if total_items > 0 else np.nan,
        })

    details_df = pd.DataFrame(rows)
    overall_acc = (total_correct_all / total_items_all) if total_items_all > 0 else float("nan")
    return details_df, overall_acc


def main():
    # Load master
    if not Path(MASTER_FILE).exists():
        raise FileNotFoundError(f"Master file not found: {MASTER_FILE}")
    master_df = pd.read_excel(MASTER_FILE)

    # Prepare writer
    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        summary_rows = []
        for child_file in CHILD_FILES:
            if not Path(child_file).exists():
                print(f"WARNING: Child file not found, skipping: {child_file}")
                continue
            child_df = pd.read_excel(child_file)

            details_df, overall_acc = score_child_against_master(master_df, child_df, TOL)

            # Write details sheet for this child
            sheet_name = Path(child_file).stem[:31]  # Excel sheet name limit
            details_df.to_excel(writer, sheet_name=sheet_name, index=False)

            summary_rows.append({
                "Child_File": child_file,
                "Accuracy_%": round(100.0 * overall_acc, 2) if not math.isnan(overall_acc) else np.nan
            })

        # Write summary
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"Done. Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
