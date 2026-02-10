import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from logger import Logger  # Import logger

class Environment:
    def __init__(self):
        self.grid_size = 10
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.obstacles = [
            [(2, 5), (5, 5), (5, 5.5), (2, 5.5)],
            [(3, 3), (3.5, 3), (3.5, 4), (3, 4)],
            [(6, 2), (8, 2), (8, 3), (6, 3)],
            [(7.5, 2), (8, 2), (8, 4), (7.5, 4)]
        ]

        for obs in self.obstacles:
            x_min = int(min(x for x, y in obs))
            x_max = int(max(x for x, y in obs))
            y_min = int(min(y for x, y in obs))
            y_max = int(max(y for x, y in obs))
            self.grid[y_min:y_max+1, x_min:x_max+1] = 1

        self.agent_pos = (1, 1)
        self.goal_pos = (6, 4)

        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = Logger(self.log_dir)
        self.logger.log("Environment initialized.")

    def create_environment(self):
        self.logger.log("Environment grid, agent start, goal, and obstacles returned.")
        return self.grid, self.agent_pos, self.goal_pos, self.obstacles

    def set_agent_and_goal(self, agent_pos, goal_pos):
        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
        self.logger.log(f"Agent position set to {agent_pos}, Goal position set to {goal_pos}")

    def set_obstacles(self, obstacle_coords):
        for (x, y) in obstacle_coords:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[y, x] = 1
        self.logger.log(f"Custom obstacles set: {obstacle_coords}")

    def save_environment(self):
        fig, ax = plt.subplots()
        ax.imshow(self.grid, cmap='Greys', origin='lower')

        if self.agent_pos:
            ax.plot(self.agent_pos[0], self.agent_pos[1], 'bo', label="Agent")
        if self.goal_pos:
            ax.plot(self.goal_pos[0], self.goal_pos[1], 'go', label="Goal")

        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.grid(True)
        ax.legend()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(self.log_dir, f"environment_{timestamp}.png")
        plt.savefig(file_path)
        plt.close()

        self.logger.log(f"Environment image saved to {file_path}")
        return file_path

