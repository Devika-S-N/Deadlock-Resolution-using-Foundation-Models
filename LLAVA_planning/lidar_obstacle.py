# lidar_obstacle.py

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import os

class LidarObstacleDetector:
    def __init__(
        self,
        occupancy_grid,
        obstacles,
        resolution=0.1,
        range_limit=1.0,
        angle_increment=np.deg2rad(10),
        logger=None
    ):
        self.grid = occupancy_grid
        self.obstacles = obstacles
        self.resolution = resolution
        self.range_limit = range_limit
        self.angle_increment = angle_increment
        self.logger = logger  # <-- shared logger from controller


    def scan(self, position):
        if self.logger:
            self.logger.log(
                f"Starting LIDAR scan at position "
                f"{position.tolist() if hasattr(position,'tolist') else position}"
            )

        hits = []
        pos_pt = Point(float(position[0]), float(position[1]))

        for angle in np.arange(0, 2 * np.pi, self.angle_increment):
            ray_end = np.array([
                position[0] + self.range_limit * np.cos(angle),
                position[1] + self.range_limit * np.sin(angle)
            ])
            ray = LineString([pos_pt, (ray_end[0], ray_end[1])])

            closest_point = None
            min_dist = float("inf")

            for obs in self.obstacles:
                if not ray.intersects(obs):
                    continue

                inter = ray.intersection(obs)
                if inter.is_empty:
                    continue

                # Normalize intersection geometry
                if inter.geom_type == "MultiPoint":
                    pts = list(inter.geoms)
                elif inter.geom_type == "Point":
                    pts = [inter]
                elif inter.geom_type in ("LineString", "MultiLineString"):
                    pts = [inter.interpolate(0.0)]
                else:
                    continue

                for p in pts:
                    d = p.distance(pos_pt)
                    if d < min_dist:
                        min_dist = d
                        closest_point = np.array([p.x, p.y])

            if closest_point is not None:
                hits.append((angle, closest_point))
                if self.logger:
                    self.logger.log(
                        f"Ray {np.rad2deg(angle):.1f}° hit at {closest_point.tolist()}"
                    )

        if self.logger:
            self.logger.log("LIDAR scan completed.")

        return hits


    def visualize(self, position, hits, goal):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_title("LIDAR Scan Visualization")

        for obs in self.obstacles:
            rect = Polygon(obs)
            x, y = rect.exterior.xy
            ax.fill(x, y, color='black')

        ax.plot(position[0], position[1], 'go', markersize=8, label="Agent")
        ax.plot(goal[0], goal[1], 'b*', markersize=12, label="Goal")

        for angle, end in hits:
            ray_end = end
            ax.plot([position[0], ray_end[0]], [position[1], ray_end[1]], color='green')

            expected_end = np.array([
                position[0] + self.range_limit * np.cos(angle),
                position[1] + self.range_limit * np.sin(angle)
            ])

            if not np.allclose(ray_end, expected_end, atol=1e-2):
                ax.plot(ray_end[0], ray_end[1], 'rx')

        ax.legend()
        plt.grid(True)
        plt.tight_layout()

        file_path = "logs/lidar_scan.png"
        plt.savefig(file_path)
        plt.close()
        if self.logger:
            self.logger.log(f"LIDAR scan visualization saved to {file_path}")
