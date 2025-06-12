from environment_fixed import Environment

# Create fixed environment
env = Environment(grid_size=(10, 10))
env.set_agent(1, 1)
env.set_goal(8, 8)
env.add_obstacle(2, 3, 4, 1)
env.add_obstacle(6, 1, 1, 3)
env.add_obstacle(2, 8, 1, 0.5)
env.add_obstacle(2, 7, 0.5, 1)
# Render and save image
env.render_image('fixed_env.png')

# create random environment
env_data = env.get_env_data()
print(env_data)

env_random = Environment(grid_size=(10, 10))
env_random.set_agent(1, 1)
env_random.set_goal(8, 8)
env_random.generate_random_obstacles(n=4, max_size=(2, 2))
env_random.render_image('random_env.png')
print(env_random.get_env_data())
