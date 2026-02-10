from shapely.geometry import LineString, box, Point

def is_inside_obstacle(x, y, obstacles):
    """Check if a point is inside any obstacle"""
    for ox, oy, w, h in obstacles:
        if ox <= x <= ox + w and oy <= y <= oy + h:
            return True
    return False

def line_intersects_obstacle(p1, p2, obstacle):
    """Check if line segment intersects with an obstacle"""
    line = LineString([p1, p2])
    ox, oy, w, h = obstacle
    rect = box(ox, oy, ox + w, oy + h)
    return line.intersects(rect)

def get_relevant_obstacles(p1, p2, all_obstacles, buffer=0.5):
    """Get obstacles near a line segment"""
    line = LineString([p1, p2])
    area = line.buffer(buffer)
    return [obs for obs in all_obstacles if box(obs[0], obs[1], obs[0] + obs[2], obs[1] + obs[3]).intersects(area)]

def waypoint_clear_of_obstacle(x, y, obstacle, buffer=0.5):
    """Check if waypoint maintains safe distance from obstacle"""
    ox, oy, w, h = obstacle
    expanded = box(ox - buffer, oy - buffer, ox + w + buffer, oy + h + buffer)
    return not expanded.contains(Point(x, y))