# main_gpt5.py
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import pandas as pd
import time   # <-- add this

from extract_scene_gpt import extract_scene_from_image as extract_gpt

IMG_DIR = Path("new_minor_grid_off")
MAX_PARALLEL_IMAGES = 4  # <-- tune here

def _fmt_xy(xy): return f"({float(xy[0]):.2f},{float(xy[1]):.2f})"
def _fmt_vertices(verts): return "[" + ", ".join(_fmt_xy(v) for v in verts) + "]"

def _write_excel(rows: List[Dict], out_path: Path):
    max_obs = 0
    for r in rows:
        for k in r.keys():
            if k.startswith("Obstacle"):
                idx = int(k.replace("Obstacle", ""))
                max_obs = max(max_obs, idx)
    cols = ["Image", "Model", "Agent", "Goal"] + [f"Obstacle{i}" for i in range(1, max_obs + 1)]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df.to_excel(out_path, index=False)

def _task(img: Path) -> Dict:
    scene = extract_gpt(str(img), model="gpt-5")
    row = {"Image": img.name, "Model": "gpt-5",
           "Agent": _fmt_xy(scene["agent"]), "Goal": _fmt_xy(scene["goal"])}
    for i, ob in enumerate(scene.get("obstacles", []), start=1):
        row[f"Obstacle{i}"] = _fmt_vertices(ob["vertices"])
    return row

def main():
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {IMG_DIR.resolve()}")
    images = sorted(IMG_DIR.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images in {IMG_DIR.resolve()}")

    rows: List[Dict] = []
    print(f"GPT-5: processing {len(images)} images with concurrency={MAX_PARALLEL_IMAGES}")

    start_time = time.perf_counter()   # <-- start timer

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as pool:
        futures = {pool.submit(_task, img): img for img in images}
        for fut in as_completed(futures):
            img = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"[OK] {img.name}")
            except Exception as e:
                rows.append({"Image": img.name, "Model": "gpt-5", "Agent": "ERR", "Goal": f"ERR: {e}"})
                print(f"[ERR] {img.name} -> {e}")

    elapsed = (time.perf_counter() - start_time)/60   # <-- stop timer
    print(f"\nTotal execution time: {elapsed:.2f} seconds")

    out_path = IMG_DIR / "results_gpt_gpt-5.xlsx"
    _write_excel(rows, out_path)
    print("[OK] Wrote:", out_path.resolve())

if __name__ == "__main__":
    main()
