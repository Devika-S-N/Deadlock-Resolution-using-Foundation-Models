import os
import pandas as pd
from global_planner import GlobalPlanner
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
    # Setup logging - creates timestamped log directory
    logger, log_dir = setup_logger()
    logger.info("=== Starting Integrated Path Planning Session ===")
    
    # Configure environment
    setup_environment()
    logger.debug("AWS environment configured")

    # Problem definition
    agent = [1, 1]
    goal = [8, 8]
    obstacles = [(3, 3, 0.5, 1), (6, 1, 2, 1), (2, 5, 3, 0.5), (7, 2, 1, 2)]
    grid_size = (10, 10)
    
    logger.info(f"\nTest Parameters:")
    logger.info(f"Agent Start: {agent}")
    logger.info(f"Goal: {goal}")
    logger.info(f"Obstacles: {obstacles}")
    logger.info(f"Grid Size: {grid_size}")

    # Initialize components
    llm_interface = LLMInterface()
    global_planner = GlobalPlanner(llm_interface)
    local_planner = LocalPlanner(llm_interface)
    
    # Global path planning
    logger.info("\n=== Starting Global Path Planning ===")
    try:
        initial_waypoints = global_planner.plan_global_path(agent, goal, obstacles, grid_size)
        full_path = [agent] + initial_waypoints + [goal]
        logger.info(f"Obtained {len(initial_waypoints)} global waypoints")
        logger.info(f"Full initial path: {full_path}")
    except Exception as e:
        logger.error(f"Global planning failed: {str(e)}")
        raise

    # Local path refinement
    logger.info("\n=== Starting Local Path Refinement ===")
    logger.info("Checking for obstacle intersections in initial path...")
    
    try:
        refined_path = local_planner.refine_global_path(full_path, obstacles)
        logger.info(f"\nRefinement Results:")
        logger.info(f"Initial waypoint count: {len(full_path)}")
        logger.info(f"Refined waypoint count: {len(refined_path)}")
        logger.info(f"Added {len(refined_path)-len(full_path)} detour points")
        logger.info(f"Final Waypoints:\n{refined_path}")
    except Exception as e:
        logger.error(f"Path refinement failed: {str(e)}")
        raise

    # Save outputs
    try:
        # Visualization
        image_path = os.path.join(log_dir, "integrated_path_planning.png")
        visualize_path(
            agent, goal, refined_path, obstacles, grid_size,
            save_path=image_path,
            original_waypoints=full_path
        )
        logger.info(f"\nVisualization saved to {image_path}")

        # Data export
        data_path = os.path.join(log_dir, "planning_results.xlsx")
        data = {
            "Agent Start": [agent],
            "Goal": [goal],
            "Obstacles": [obstacles],
            "Initial Waypoints": [full_path],
            "Refined Waypoints": [refined_path]
        }
        pd.DataFrame(data).to_excel(data_path, index=False)
        logger.info(f"Test data saved to {data_path}")

        logger.info("\n=== Integrated Path Planning Completed Successfully ===")
        logger.info(f"All outputs saved in directory: {log_dir}")
    except Exception as e:
        logger.error(f"Output saving failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()