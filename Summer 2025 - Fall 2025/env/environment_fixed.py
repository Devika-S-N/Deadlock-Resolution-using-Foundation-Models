import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class Environment:
    def __init__(self, grid_size=(10, 10)):
        self.grid_size = grid_size
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = []

    def set_agent(self, x, y):
        self.agent_pos = [x, y]

    def set_goal(self, x, y):
        self.goal_pos = [x, y]

    def add_obstacle(self, x, y, width, height):
        self.obstacles.append((x, y, width, height))

    def render_image(self, save_path='env.png'):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_aspect('equal')
        ax.grid(True)

        # Draw agent
        if self.agent_pos:
            ax.plot(self.agent_pos[0] + 0.5, self.agent_pos[1] + 0.5, 'bo', markersize=10, label='Agent')

        # Draw goal
        if self.goal_pos:
            ax.plot(self.goal_pos[0] + 0.5, self.goal_pos[1] + 0.5, 'go', markersize=10, label='Goal')

        # Draw obstacles
        for obs in self.obstacles:
            rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3], linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)

        ax.legend()
        plt.savefig(save_path)
        plt.close()

    def get_env_data(self):
        return [{
            'agent': self.agent_pos,
            'goal': self.goal_pos,
            'obstacles': self.obstacles
        }]

    def generate_random_obstacles(self, n=5, max_size=(2, 2)):
        for _ in range(n):
            width = random.randint(1, max_size[0])
            height = random.randint(1, max_size[1])
            x = random.randint(0, self.grid_size[0] - width)
            y = random.randint(0, self.grid_size[1] - height)
            self.add_obstacle(x, y, width, height)
