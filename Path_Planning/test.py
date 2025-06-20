import os
import pandas as pd
from local_planner import LocalPlanner
from utils.llm_interface import LLMInterface
from utils.visualization import visualize_path
from utils.logger import setup_logger
from utils.geometry import line_intersects_obstacle

def setup_environment():
    """Configure AWS settings"""
    os.environ['AWS_PROFILE'] = "MyProfile1"
    os.environ['AWS_DEFAULT_REGION'] = "us-east-1"

def test_local_planner():
    # Setup logging - creates timestamped log directory
    logger, log_dir = setup_logger()
    logger.info("=== Starting Local Planner Test ===")
    
    # Configure environment
    setup_environment()
    logger.debug("AWS environment configured")

    # Test case definition
    agent = [1, 1]
    goal = [9, 2]
    initial_waypoints = [
        [1, 1], [2, 2], [3, 2], [4, 2], 
        [5, 3], [6.5, 3], [8.5, 3], [9, 2]
    ]
    obstacles = [
        (3, 3, 0.5, 1),  # This obstacle intersects with the path
        (6, 1, 2, 1), 
        (2, 5, 3, 0.5), 
        (7, 2, 1, 2)
    ]
    grid_size = (10, 10)
    
    logger.info(f"\nTest Parameters:")
    logger.info(f"Agent Start: {agent}")
    logger.info(f"Goal: {goal}")
    logger.info(f"Initial Waypoints: {initial_waypoints}")
    logger.info(f"Obstacles: {obstacles}")
    logger.info(f"Grid Size: {grid_size}")

    # Initialize components
    llm_interface = LLMInterface()
    local_planner = LocalPlanner(llm_interface)

    logger.info("\n=== Testing Path Refinement ===")
    logger.info("Checking for obstacle intersections in initial path...")
    
    # Process the initial path
    try:
        refined_path = local_planner.refine_global_path(initial_waypoints, obstacles)
        logger.info(f"\nRefinement Results:")
        logger.info(f"Initial waypoint count: {len(initial_waypoints)}")
        logger.info(f"Refined waypoint count: {len(refined_path)}")
        logger.info(f"Added {len(refined_path)-len(initial_waypoints)} detour points")
        logger.info(f"Final Waypoints:\n{refined_path}")
    except Exception as e:
        logger.error(f"Path refinement failed: {str(e)}")
        raise

    # Save outputs
    try:
        # Visualization
        image_path = os.path.join(log_dir, "local_planner_test.png")
        visualize_path(
            agent, goal, refined_path, obstacles, grid_size,
            save_path=image_path,
            original_waypoints=initial_waypoints
        )
        logger.info(f"\nVisualization saved to {image_path}")

        # Data export
        data_path = os.path.join(log_dir, "test_results.xlsx")
        data = {
            "Initial Waypoints": [initial_waypoints],
            "Refined Waypoints": [refined_path],
            "Obstacles": [obstacles],
            "Agent": [agent],
            "Goal": [goal]
        }
        pd.DataFrame(data).to_excel(data_path, index=False)
        logger.info(f"Test data saved to {data_path}")

        logger.info("\n=== Local Planner Test Completed Successfully ===")
        logger.info(f"All outputs saved in directory: {log_dir}")
    except Exception as e:
        logger.error(f"Output saving failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_local_planner()