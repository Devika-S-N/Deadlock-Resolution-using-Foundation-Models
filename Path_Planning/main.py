import os
import pandas as pd
from global_planner_1 import GlobalPlanner
from local_planner import LocalPlanner
from utils.llm_interface import LLMInterface
from utils.geometry import line_intersects_obstacle
from utils.visualization import visualize_path
from utils.logger import setup_logger

def setup_environment():
    """Configure AWS settings"""
    os.environ['AWS_PROFILE'] = "MyProfile1"
    os.environ['AWS_DEFAULT_REGION'] = "us-east-1"

def main():
    # Setup logging - this creates the timestamped log directory
    logger, log_dir = setup_logger()
    logger.info("=== Starting Path Planning Session ===")
    
    # Configure environment
    setup_environment()
    logger.debug("AWS environment configured")

    # Problem definition
    agent = [1, 1]
    goal = [8, 8]
    obstacles = [(3, 3, 0.5, 1), (6, 1, 2, 1), (2, 5, 3, 0.5), (7, 2, 1, 2)]
    grid_size = (10, 10)
    
    logger.info(f"Agent: {agent}, Goal: {goal}")
    logger.info(f"Grid size: {grid_size}")
    logger.info(f"Obstacles: {obstacles}")


    # Initialize components
    llm_interface = LLMInterface()
    global_planner = GlobalPlanner(llm_interface)
    local_planner = LocalPlanner(llm_interface)
    
    # Global path planning
    logger.info(f"\n********************************************************************************************************************************\n")

    logger.info("=== Starting global path planning ===")
    try:
        waypoints = global_planner.plan_global_path(agent, goal, obstacles, grid_size)
        logger.info(f"Obtained {len(waypoints)} global waypoints: {waypoints}")
    except Exception as e:
        logger.error(f"Global planning failed: {str(e)}")
        raise

    # Check for obstacles and plan local detours
    path = [agent] + waypoints + [goal]
    final_path = [agent]
    





    # Save all outputs to the log directory
    try:
        # Visualization
        image_path = os.path.join(log_dir, "path_visualization.png")


        visualize_path(agent, goal, waypoints, obstacles, grid_size, save_path=image_path, original_waypoints=waypoints)

        logger.info(f"Visualization saved to {image_path}")

        # Data export
        data_path = os.path.join(log_dir, "path_data.xlsx")
        data = {
            "Agent Start": [agent],
            "Goal": [goal],
            "Obstacles": [obstacles],
            "Final Waypoints": [final_path]
        }
        pd.DataFrame(data).to_excel(data_path, index=False)
        logger.info(f"Data saved to {data_path}")

        logger.info("=== Path Planning Completed Successfully ===")
        logger.info(f"All outputs saved in directory: {log_dir}")
    except Exception as e:
        logger.error(f"Output saving failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()


'''    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        segment_clear = True
        
        for obs in obstacles:
            if line_intersects_obstacle(p1, p2, obs):
                logger.warning(f"Obstacle detected between {p1} and {p2}")
                try:
                    detour = local_planner.plan_local_path(p1, p2, obstacles)
                    logger.info(f"Generated {len(detour)} detour waypoints: {detour}")
                    final_path.extend(detour)
                    segment_clear = False
                    break
                except Exception as e:
                    logger.error(f"Local planning failed: {str(e)}")
                    raise
                
        if segment_clear:
            final_path.append(p2)'''






'''
        visualize_path(agent, goal, final_path, obstacles, grid_size,
                     save_path=image_path,
                     original_waypoints=waypoints)'''