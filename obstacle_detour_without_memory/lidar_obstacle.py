import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from environment import Environment
from logger import Logger  # New import
import os

class LidarObstacleDetector:
    def __init__(self, occupancy_grid, obstacles, resolution=0.1, range_limit=0.5, angle_increment=np.deg2rad(10)):
        self.grid = occupancy_grid
        self.obstacles = obstacles
        self.resolution = resolution
        self.range_limit = range_limit
        self.angle_increment = angle_increment

        os.makedirs("logs", exist_ok=True)
        self.logger = Logger("logs")
        self.logger.log("Initialized LidarObstacleDetector.")

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

        if intersection is not None:
            self.logger.log(f"Ray hit at {intersection}")
        return (intersection is not None), intersection

    def scan(self, position):
        self.logger.log(f"Starting LIDAR scan at position {position}")
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
                self.logger.log(f"Ray at angle {np.rad2deg(angle):.1f}° hit obstacle at {closest_point}")
            else:
                hits.append((angle, ray_end))
                self.logger.log(f"Ray at angle {np.rad2deg(angle):.1f}° hit nothing.")

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
        self.logger.log(f"LIDAR scan visualization saved to {file_path}")


'''if __name__ == "__main__":
    env = Environment()
    grid, agent_pos, goal_pos, obstacles = env.create_environment()

    lidar = LidarObstacleDetector(grid, obstacles)
    hits = lidar.scan(agent_pos)
    lidar.visualize(agent_pos, hits, goal_pos)
'''




'''# lidar_obstacle.py

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from environment import Environment
import os

class LidarObstacleDetector:
    def __init__(self, occupancy_grid, obstacles, resolution=0.1, range_limit=0.5, angle_increment=np.deg2rad(30)):
        self.grid = occupancy_grid
        self.obstacles = obstacles
        self.resolution = resolution
        self.range_limit = range_limit
        self.angle_increment = angle_increment

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

        return (intersection is not None), intersection

    def scan(self, position):
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
            else:
                hits.append((angle, ray_end))

        return hits

    def visualize(self, position, hits, goal):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_title("LIDAR Scan Visualization")

        # Plot obstacles
        for obs in self.obstacles:
            rect = Polygon(obs)
            x, y = rect.exterior.xy
            ax.fill(x, y, color='black')

        # Plot agent
        ax.plot(position[0], position[1], 'go', markersize=8, label="Agent")

        # Plot goal
        ax.plot(goal[0], goal[1], 'b*', markersize=12, label="Goal")

        # Plot rays and mark only obstacle hits with 'x'
        for angle, end in hits:
            ray_end = end
            ax.plot([position[0], ray_end[0]], [position[1], ray_end[1]], color='green')

            # Recompute expected free ray end for this angle
            expected_end = np.array([
                position[0] + self.range_limit * np.cos(angle),
                position[1] + self.range_limit * np.sin(angle)
            ])

            # If actual end is closer than expected (i.e., hit happened), plot red x
            if not np.allclose(ray_end, expected_end, atol=1e-2):
                ax.plot(ray_end[0], ray_end[1], 'rx')  # Only mark hits

        ax.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("logs", exist_ok=True)
        plt.savefig("logs/lidar_scan.png")
        plt.show()


if __name__ == "__main__":
    env = Environment()
    grid, agent_pos, goal_pos, obstacles = env.create_environment()

    lidar = LidarObstacleDetector(grid, obstacles)
    hits = lidar.scan(agent_pos)
    lidar.visualize(agent_pos, hits, goal_pos)
'''