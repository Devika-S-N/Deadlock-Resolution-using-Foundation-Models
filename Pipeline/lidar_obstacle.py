# lidar_obstacle.py

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
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

    def ray_intersection(self, ray_start: np.ndarray, ray_end: np.ndarray, polygon: Polygon):
        x1, y1 = ray_start
        x2, y2 = ray_end
        coords = list(polygon.exterior.coords)

        min_alpha = float('inf')
        intersection = None

        for i in range(len(coords) - 1):
            x3, y3 = coords[i]
            x4, y4 = coords[i + 1]

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue

            alpha = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            beta = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

            if 0 <= alpha <= 1 and 0 <= beta <= 1:
                if alpha < min_alpha:
                    min_alpha = alpha
                    intersection = np.array([x1 + alpha * (x2 - x1), y1 + alpha * (y2 - y1)])

        # if intersection is not None and self.logger:
        #     self.logger.log(f"Ray hit at {intersection.tolist()}")
        return (intersection is not None), intersection

    def scan(self, position):
        if self.logger:
            self.logger.log(f"Starting LIDAR scan at position {position.tolist() if hasattr(position,'tolist') else position}")

        hits = []
        for angle in np.arange(0, 2 * np.pi, self.angle_increment):
            x_end = position[0] + self.range_limit * np.cos(angle)
            y_end = position[1] + self.range_limit * np.sin(angle)
            ray_end = np.array([x_end, y_end])

            closest_point = None
            min_dist = float('inf')

            for obs in self.obstacles:
                polygon = Polygon(obs)
                hit, point = self.ray_intersection(position, ray_end, polygon)
                if hit:
                    dist = np.linalg.norm(point - position)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = point

            if closest_point is not None:
                hits.append((angle, closest_point))
                if self.logger:
                    self.logger.log(f"Ray at angle {np.rad2deg(angle):.1f}° hit obstacle at {closest_point.tolist()}")
            # else:
            #     hits.append((angle, ray_end, False))
            #     # optional: log free rays if you want
            #     # if self.logger:
            #     #     self.logger.log(f"Ray at angle {np.rad2deg(angle):.1f}° free to {ray_end.tolist()}")

        if self.logger:
            self.logger.log("LIDAR scan completed.")
        return hits # list of (angle, hit_point) — only real hits

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
