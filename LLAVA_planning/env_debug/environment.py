import json
import numpy as np
from warehouse import generate_env
from shapely.geometry import LineString
import matplotlib.pyplot as plt


def walls_from_warehouse(vwall, hwall, N):
    walls = []
    for col in range(N + 1):
        for r in range(N):
            if vwall[col][r]:
                y0 = N - (r + 1)
                y1 = N - r
                walls.append(LineString([(col, y0), (col, y1)]))

    for row in range(N + 1):
        for c in range(N):
            if hwall[row][c]:
                y = N - row
                walls.append(LineString([(c, y), (c + 1, y)]))
    return walls


def _cell_center_xy(r, c, N):
    return (float(c), float(N - r))


def _rect_poly_from_cells(r0, c0, h, w, N):
    x0 = float(c0)
    x1 = float(c0 + w)
    y_top = float(N - r0)
    y_bot = float(N - (r0 + h))
    return [(x0, y_bot), (x1, y_bot), (x1, y_top), (x0, y_top)]


class Environment:
    def __init__(
        self,
        N=15,
        num_rooms=5,
        num_obstacles=10,
        seed=0,
        num_images=1,
        out_dir=".",
        add_workstations=False,
        difficulty="medium",
    ):
        self.N = N
        self.num_rooms = num_rooms
        self.num_obstacles = num_obstacles
        self.seed = seed
        self.num_images = num_images
        self.out_dir = out_dir
        self.add_workstations = add_workstations
        self.difficulty = difficulty

    def create_environment(self):
        envs = []

        for i in range(self.num_images):
            cur_seed = self.seed if self.num_images == 1 else self.seed + i

            out_png, out_json = generate_env(
                N=self.N,
                num_rooms=self.num_rooms,
                num_obstacles=self.num_obstacles,
                seed=cur_seed,
                out_dir=self.out_dir,
                add_workstations=self.add_workstations,
                difficulty=self.difficulty,
            )

            with open(out_json, "r") as f:
                meta = json.load(f)

            N = meta["N"]
            ar, ac = meta["agent"]["r"], meta["agent"]["c"]
            gr, gc = meta["goal"]["r"], meta["goal"]["c"]

            agent_xy = _cell_center_xy(ar, ac, N)
            goal_xy = _cell_center_xy(gr, gc, N)

            obstacles = [
                _rect_poly_from_cells(o["r0"], o["c0"], o["h"], o["w"], N)
                for o in meta["obstacles"]
            ]
            obstacles += walls_from_warehouse(meta["vwall_room"], meta["hwall_room"], N)

            envs.append(
                {
                    "seed": cur_seed,
                    "png": out_png,
                    "agent": agent_xy,
                    "goal": goal_xy,
                    "obstacles": obstacles,
                }
            )

        return envs


if __name__ == "__main__":
    print("Running environment.py standalone")

    env = Environment(
        N=15,
        num_rooms=3,
        num_obstacles=10,
        seed=0,
        num_images=50,
        out_dir=".",
        difficulty="medium",
    )

    envs = env.create_environment()
    for e in envs:
        print("Seed:", e["seed"], "PNG:", e["png"])

