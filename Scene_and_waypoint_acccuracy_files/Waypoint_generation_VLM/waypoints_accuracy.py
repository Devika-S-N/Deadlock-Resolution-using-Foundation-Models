#!/usr/bin/env python3
"""
waypoints_accuracy.py

Compute waypoint validity metrics from model result sheets.
"""

import re
import ast
import math
from pathlib import Path
from typing import Any, List, Tuple, Dict
import pandas as pd
import numpy as np

# ---------- config ----------
CHILD_FILES = [
    "results_aws_haiku.xlsx",
    "results_aws_sonnet.xlsx",
    "results_gpt_gpt-4o.xlsx",
    "results_gpt_gpt-4o-mini.xlsx",
    "results_gpt_gpt-5.xlsx",
]
OUTPUT_FILE = "waypoints_accuracy.xlsx"

COL_IMAGE      = "image"
COL_MODEL      = "model"
COL_WAYPOINTS  = "waypoints"
COL_INV_POINTS = "invalid points"
COL_SEG_INT    = "segment intersect"

# ---------- helpers ----------
def _norm(s: str) -> str:
    return "".join(str(s).strip().lower().split())

def resolve_cols(df: pd.DataFrame, logical: List[str]) -> Dict[str, str]:
    actuals = { _norm(c): c for c in df.columns }
    out = {}
    for name in logical:
        n = _norm(name)
        if n in actuals:
            out[name] = actuals[n]
    return out

def _safe_literal_eval(s: Any):
    if s is None or (isinstance(s, float) and math.isnan(s)): return None
    if isinstance(s, (list, tuple, dict)): return s
    if isinstance(s, str):
        txt = s.strip()
        if txt == "" or txt.lower() in {"none", "nan"}: return None
        try:
            return ast.literal_eval(txt)
        except Exception:
            if txt.startswith("(") and txt.endswith(")"):
                try:
                    return ast.literal_eval(f"[{txt}]")
                except Exception:
                    return None
            return None
    return None

def _to_point_list(val: Any) -> List[Tuple[float,float]]:
    parsed = _safe_literal_eval(val)
    if parsed is None: return []
    pts: List[Tuple[float,float]] = []
    def _as_xy(p):
        if isinstance(p, (list, tuple)) and len(p)==2:
            try: return (float(p[0]), float(p[1]))
            except Exception: return None
        return None
    if isinstance(parsed, (list, tuple)):
        if len(parsed)==2 and all(isinstance(c,(int,float)) for c in parsed):
            xy = _as_xy(parsed)
            if xy: pts.append(xy)
        else:
            for it in parsed:
                xy = _as_xy(it)
                if xy: pts.append(xy)
    return pts

SEG_PAIR_RE = re.compile(r"\(\s*([0-9.+-]+)\s*,\s*([0-9.+-]+)\s*\)\s*-\s*\(\s*([0-9.+-]+)\s*,\s*([0-9.+-]+)\s*\)")

def _parse_segment_pairs(val: Any) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
    if val is None or (isinstance(val, float) and math.isnan(val)): return []
    s = str(val)
    pairs = []
    for m in SEG_PAIR_RE.finditer(s):
        x1, y1, x2, y2 = m.groups()
        try:
            a = (float(x1), float(y1))
            b = (float(x2), float(y2))
            pairs.append((a,b))
        except Exception:
            continue
    return pairs

def _scores_for_row(waypoints, invalid_pts, bad_segs):
    n_w = len(waypoints)
    n_i = len(invalid_pts)
    n_s = len(bad_segs)

    point_score = (n_w - n_i) / n_w if n_w > 0 else np.nan
    n_seg = max(n_w - 1, 0)
    segment_score = (n_seg - n_s) / n_seg if n_seg > 0 else (np.nan if n_w==0 else 1.0)

    if math.isnan(point_score) and math.isnan(segment_score):
        combined = np.nan
    elif math.isnan(point_score):
        combined = segment_score
    elif math.isnan(segment_score):
        combined = point_score
    else:
        combined = 0.5*point_score + 0.5*segment_score

    return {
        "N_waypoints": n_w,
        "N_invalid_points": n_i,
        "N_bad_segments": n_s,
        "Point_Score": point_score,
        "Segment_Score": segment_score,
        "Combined_Accuracy": combined,
    }

# ---------- core ----------
def process_file(child_path: Path) -> pd.DataFrame:
    print(f"[INFO] Processing {child_path.name}")
    df = pd.read_excel(child_path)

    cols = resolve_cols(df, [
        COL_IMAGE, COL_MODEL, COL_WAYPOINTS, COL_INV_POINTS, COL_SEG_INT
    ])
    rows = []
    for _, r in df.iterrows():
        wps   = _to_point_list(r[cols[COL_WAYPOINTS]])
        inv   = _to_point_list(r[cols[COL_INV_POINTS]])
        segs  = _parse_segment_pairs(r[cols[COL_SEG_INT]])

        scores = _scores_for_row(wps, inv, segs)
        out = {
            "Image": r[cols[COL_IMAGE]],
            "Model": r[cols[COL_MODEL]],
            **scores
        }
        rows.append(out)

    return pd.DataFrame(rows)

def main():
    all_dfs = []
    summary = []

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        for fname in CHILD_FILES:
            p = Path(fname)
            if not p.exists():
                print(f"[WARN] Skipping missing file: {fname}")
                continue

            details_df = process_file(p)
            if details_df.empty:
                print(f"[WARN] No rows in {fname}")
                continue

            mean_combined = details_df["Combined_Accuracy"].mean(skipna=True)
            mean_point    = details_df["Point_Score"].mean(skipna=True)
            mean_segment  = details_df["Segment_Score"].mean(skipna=True)

            # write details
            sheet = p.stem[:31]
            details_df.to_excel(writer, sheet_name=sheet, index=False)

            summary.append({
                "Model_File": fname,
                "Mean_Combined_Accuracy": round(mean_combined, 4),
                "Mean_Point_Score": round(mean_point, 4),
                "Mean_Segment_Score": round(mean_segment, 4),
                "Combined_Accuracy_%": round(mean_combined * 100, 2),
                "Point_Score_%": round(mean_point * 100, 2),
                "Segment_Score_%": round(mean_segment * 100, 2),
                "Images_Evaluated": len(details_df),
                "Total_Waypoints": int(details_df["N_waypoints"].sum()),
                "Total_Invalid_Points": int(details_df["N_invalid_points"].sum()),
                "Total_Bad_Segments": int(details_df["N_bad_segments"].sum()),
            })

        if summary:
            summ_df = pd.DataFrame(summary).sort_values("Mean_Combined_Accuracy", ascending=False)
            # add rank column
            summ_df.insert(0, "Rank", range(1, len(summ_df) + 1))
            summ_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"[OK] Wrote output Excel: {Path(OUTPUT_FILE).resolve()}")


if __name__ == "__main__":
    main()
