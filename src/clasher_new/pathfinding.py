"""Coarse-grid A* pathfinding, used to route ground troops around buildings
instead of the old direct-line + 16-angle-scan fallback in battle.py.

Kept standalone (no BattleState/Entity imports) so it's testable without spinning
up a full battle - it only needs a walkability test, a list of (x, y, radius)
obstacles, and start/goal points.
"""
import heapq
import math
from core import Position

CELL = 0.5  # grid resolution, in tiles
NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.41421), (-1, 1, 1.41421), (1, -1, 1.41421), (1, 1, 1.41421)]


def _to_cell(pos):
    return (int(pos.x / CELL), int(pos.y / CELL))


def _cell_center(cell):
    return Position((cell[0] + 0.5) * CELL, (cell[1] + 0.5) * CELL)


def _cell_walkable(cell, gw, gh, is_walkable_fn, mover_radius, obstacles):
    if not (0 <= cell[0] < gw and 0 <= cell[1] < gh):
        return False
    pos = _cell_center(cell)
    if not is_walkable_fn(pos):
        return False
    for ox, oy, orad in obstacles:
        if pos.distance_to(Position(ox, oy)) < orad + mover_radius:
            return False
    return True


def _nearest_walkable_cell(cell, gw, gh, is_walkable_fn, mover_radius, obstacles, max_radius=6):
    if _cell_walkable(cell, gw, gh, is_walkable_fn, mover_radius, obstacles):
        return cell
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                c = (cell[0] + dx, cell[1] + dy)
                if _cell_walkable(c, gw, gh, is_walkable_fn, mover_radius, obstacles):
                    return c
    return None


def line_clear(a: Position, b: Position, is_walkable_fn, mover_radius, obstacles, step=0.2):
    """True if a straight line from a to b never enters an unwalkable tile or an obstacle's radius."""
    dist = a.distance_to(b)
    if dist == 0:
        return True
    steps = max(1, int(dist / step))
    for s in range(steps + 1):
        t = s / steps
        p = Position(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        if not is_walkable_fn(p):
            return False
        for ox, oy, orad in obstacles:
            if p.distance_to(Position(ox, oy)) < orad + mover_radius:
                return False
    return True


def _simplify(waypoints, is_walkable_fn, mover_radius, obstacles):
    """Greedy line-of-sight pruning: drop intermediate waypoints whenever a straight
    line to a further one is still clear, so troops don't zig-zag along grid cells."""
    if len(waypoints) <= 2:
        return waypoints
    simplified = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        j = len(waypoints) - 1
        while j > i + 1 and not line_clear(waypoints[i], waypoints[j], is_walkable_fn, mover_radius, obstacles):
            j -= 1
        simplified.append(waypoints[j])
        i = j
    return simplified


def find_path(start: Position, goal: Position, is_walkable_fn, mover_radius, obstacles,
              grid_width=18, grid_height=32, max_nodes=4000):
    """Returns a list of Positions from start to goal that avoids `obstacles`
    (list of (x, y, radius) tuples, already excluding whatever the caller wants
    to be allowed to walk into - e.g. the troop's own attack target).
    Falls back to [goal] directly if no path is found, so callers always get
    at least one waypoint back."""
    if line_clear(start, goal, is_walkable_fn, mover_radius, obstacles):
        return [goal]

    gw, gh = int(grid_width / CELL), int(grid_height / CELL)
    start_cell = _to_cell(start)
    goal_cell = _nearest_walkable_cell(_to_cell(goal), gw, gh, is_walkable_fn, mover_radius, obstacles)
    if goal_cell is None:
        return [goal]
    if start_cell == goal_cell:
        return [goal]

    def h(c):
        return math.hypot(c[0] - goal_cell[0], c[1] - goal_cell[1])

    open_heap = [(h(start_cell), 0.0, start_cell)]
    came_from = {}
    gscore = {start_cell: 0.0}
    visited = set()
    expanded = 0

    while open_heap and expanded < max_nodes:
        _, cur_g, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        expanded += 1
        if cur == goal_cell:
            break
        for dx, dy, cost in NEIGHBORS:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb in visited or not _cell_walkable(nb, gw, gh, is_walkable_fn, mover_radius, obstacles):
                continue
            if dx != 0 and dy != 0:
                # don't let the path cut diagonally through a blocked corner
                if not _cell_walkable((cur[0] + dx, cur[1]), gw, gh, is_walkable_fn, mover_radius, obstacles) or \
                   not _cell_walkable((cur[0], cur[1] + dy), gw, gh, is_walkable_fn, mover_radius, obstacles):
                    continue
            ng = cur_g + cost
            if ng < gscore.get(nb, float('inf')):
                gscore[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))

    if goal_cell not in came_from and goal_cell != start_cell:
        return [goal]  # no path found - let the caller's normal collision handling deal with it

    path_cells = [goal_cell]
    cur = goal_cell
    while cur != start_cell:
        cur = came_from[cur]
        path_cells.append(cur)
    path_cells.reverse()

    waypoints = [_cell_center(c) for c in path_cells]
    waypoints[-1] = goal
    return _simplify(waypoints, is_walkable_fn, mover_radius, obstacles)
