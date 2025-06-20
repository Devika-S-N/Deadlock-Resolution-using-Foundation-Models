import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box, Point

def visualize_path(agent, goal, waypoints, obstacles, grid_size, save_path, original_waypoints=None):
    """
    Visualize the path with agent, goal, obstacles, and waypoints
    
    Args:
        agent (list): Starting position [x, y]
        goal (list): Goal position [x, y]
        waypoints (list): List of waypoints [[x1,y1], [x2,y2], ...]
        obstacles (list): List of obstacles (x, y, width, height)
        grid_size (tuple): Size of the grid (width, height)
        save_path (str): Path to save the visualization
        original_waypoints (list, optional): Original waypoints before local planning
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set up grid
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_xticks(range(grid_size[0] + 1))
    ax.set_yticks(range(grid_size[1] + 1))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')
    ax.set_title("Agent Path Planning", pad=20)
    
    # Draw obstacles
    for ox, oy, w, h in obstacles:
        rect = patches.Rectangle(
            (ox, oy), w, h, 
            linewidth=1, 
            edgecolor='red', 
            facecolor='red', 
            alpha=0.5,
            label='Obstacle'
        )
        ax.add_patch(rect)
    
    # Plot agent and goal
    ax.plot(agent[0], agent[1], 'go', markersize=12, label="Start", markeredgecolor='black')
    ax.plot(goal[0], goal[1], 'b*', markersize=15, label="Goal", markeredgecolor='black')
    
    # Plot waypoints with differentiation between original and secondary
    if original_waypoints:
        original_set = set(map(tuple, original_waypoints))
        secondary_pts = [pt for pt in waypoints if tuple(pt) not in original_set and pt != agent and pt != goal]
        primary_pts = [pt for pt in waypoints if tuple(pt) in original_set]
        
        if primary_pts:
            px, py = zip(*primary_pts)
            ax.plot(px, py, 'ko', markersize=8, label="Global Waypoints", markeredgecolor='black')
        
        if secondary_pts:
            sx, sy = zip(*secondary_pts)
            ax.plot(sx, sy, 'o', color='orange', markersize=8, label="Local Waypoints", markeredgecolor='black')
    else:
        # If no original waypoints provided, plot all as primary
        if waypoints:
            wx, wy = zip(*waypoints)
            ax.plot(wx, wy, 'ko', markersize=8, label="Waypoints", markeredgecolor='black')
    
    # Draw path lines
    if waypoints:
        full_path = [agent] + waypoints + [goal] if waypoints[-1] != goal else [agent] + waypoints
        ax.plot(
            [p[0] for p in full_path],
            [p[1] for p in full_path],
            'k--', linewidth=1, alpha=0.7, label="Path"
        )
    
    # Add legend and style
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True, 
        shadow=True, 
        ncol=4
    )
    plt.tight_layout()
    
    # Save and close
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)