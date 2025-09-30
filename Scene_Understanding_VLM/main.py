# main_fast.py
# Process images sequentially; for each image, run 4 models in parallel:
#   AWS: haiku, sonnet
#   OpenAI: gpt-4o, gpt-4o-mini
# Writes:
#   parent_images_log/results_aws_haiku.xlsx
#   parent_images_log/results_aws_sonnet.xlsx
#   parent_images_log/results_gpt_gpt-4o.xlsx
#   parent_images_log/results_gpt_gpt-4o-mini.xlsx
#
# pip install pandas openpyxl

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import pandas as pd
import time
from extract_scene_aws import extract_scene_from_image as extract_aws
from extract_scene_gpt import extract_scene_from_image as extract_gpt



IMG_DIR = Path("new_minor_grid_off")

MODELS: List[Tuple[str, str]] = [
    ("aws", "haiku"),
    ("aws", "sonnet"),
    ("gpt", "gpt-4o"),
    ("gpt", "gpt-4o-mini"),
]

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

def _call_one(model_family: str, model_name: str, img: Path) -> Tuple[str, Dict]:
    caller = extract_aws if model_family == "aws" else extract_gpt
    scene = caller(str(img), model=model_name)
    row = {"Image": img.name, "Model": model_name,
           "Agent": _fmt_xy(scene["agent"]), "Goal": _fmt_xy(scene["goal"])}
    for i, ob in enumerate(scene.get("obstacles", []), start=1):
        row[f"Obstacle{i}"] = _fmt_vertices(ob["vertices"])
    return model_name, row

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
                        {"Image": img.name, "Model": name, "Agent": "ERR", "Goal": f"ERR: {e}"}
                    )
                    print(f"[ERR] {name} -> {e}")

    _write_excel(per_model_rows["haiku"],      IMG_DIR / "results_aws_haiku.xlsx")
    _write_excel(per_model_rows["sonnet"],     IMG_DIR / "results_aws_sonnet.xlsx")
    _write_excel(per_model_rows["gpt-4o"],     IMG_DIR / "results_gpt_gpt-4o.xlsx")
    _write_excel(per_model_rows["gpt-4o-mini"],IMG_DIR / "results_gpt_gpt-4o-mini.xlsx")
    print("\n[OK] Wrote 4 Excel files to:", IMG_DIR.resolve())

if __name__ == "__main__":
    start_time = time.perf_counter() 
    main()
    elapsed = (time.perf_counter() - start_time)/60   # <-- stop timer
    print(f"\nTotal execution time: {elapsed:.2f} minutes")