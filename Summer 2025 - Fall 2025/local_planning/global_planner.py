import logging
import numpy as np

class GlobalPlanner:
    def __init__(self, llm_interface=None):
        self.logger = logging.getLogger('path_planner')

    def plan_global_path(self, agent, goal, obstacles=None, grid_size=None):
        """
        Generate a straight-line global path with fixed step size (1.0 units)
        from agent to goal, without considering obstacles.
        """
        self.logger.info("Generating naive straight-line path from agent to goal")

        agent = np.array(agent)
        goal = np.array(goal)
        direction = goal - agent
        distance = np.linalg.norm(direction)

        if distance == 0:
            return []

        step_size = 1.0
        num_steps = int(distance / step_size)
        unit_vector = direction / distance

        waypoints = [list(agent + i * step_size * unit_vector) for i in range(1, num_steps)]
        return waypoints
