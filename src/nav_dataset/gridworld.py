from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


Position = Tuple[int, int]


# 8-connected actions: N, NE, E, SE, S, SW, W, NW
ACTION_DELTAS: List[Position] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]


DELTA_TO_ACTION_ID: Dict[Position, int] = {d: i for i, d in enumerate(ACTION_DELTAS)}


@dataclass(frozen=True)
class GridConfig:
    height: int = 64
    width: int = 64
    obstacle_density: float = 0.2
    min_start_goal_l2: float = 12.0
    map_type: str = "random"  # random | maze
    maze_extra_open_prob: float = 0.02


def make_random_grid(config: GridConfig, rng: np.random.Generator) -> np.ndarray:
    """Create occupancy map (1=obstacle, 0=free)."""
    grid = (rng.random((config.height, config.width)) < config.obstacle_density).astype(np.uint8)
    return grid


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n - 1


def make_maze_grid(config: GridConfig, rng: np.random.Generator) -> np.ndarray:
    """Create a maze-like occupancy map using randomized DFS (1=obstacle, 0=free)."""
    if config.height < 5 or config.width < 5:
        raise ValueError("Maze generation requires height and width >= 5")

    h = _ensure_odd(config.height)
    w = _ensure_odd(config.width)

    maze = np.ones((h, w), dtype=np.uint8)
    start = (1, 1)
    maze[start] = 0
    stack: List[Position] = [start]

    # Carve passages by jumping 2 cells and opening the wall in-between.
    while stack:
        r, c = stack[-1]
        candidates: List[Tuple[int, int, int, int]] = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr < h - 1 and 1 <= nc < w - 1 and maze[nr, nc] == 1:
                candidates.append((nr, nc, r + dr // 2, c + dc // 2))

        if not candidates:
            stack.pop()
            continue

        nr, nc, wr, wc = candidates[int(rng.integers(0, len(candidates)))]
        maze[wr, wc] = 0
        maze[nr, nc] = 0
        stack.append((nr, nc))

    # Optionally open extra random walls to add loops/alternative routes.
    if config.maze_extra_open_prob > 0.0:
        random_open = rng.random((h, w)) < config.maze_extra_open_prob
        maze[random_open] = 0

    # If requested shape is even-sized, pad with obstacle boundary.
    if h != config.height or w != config.width:
        out = np.ones((config.height, config.width), dtype=np.uint8)
        out[:h, :w] = maze
        return out
    return maze


def make_grid(config: GridConfig, rng: np.random.Generator) -> np.ndarray:
    if config.map_type == "random":
        return make_random_grid(config, rng)
    if config.map_type == "maze":
        return make_maze_grid(config, rng)
    raise ValueError(f"Unsupported map_type='{config.map_type}'. Use 'random' or 'maze'.")


def in_bounds(grid: np.ndarray, pos: Position) -> bool:
    r, c = pos
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]


def is_free(grid: np.ndarray, pos: Position) -> bool:
    r, c = pos
    return grid[r, c] == 0


def heuristic(a: Position, b: Position) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    # Octile distance for 8-connected grid with cardinal cost=1, diagonal cost=sqrt(2)
    return (max(dr, dc) - min(dr, dc)) + math.sqrt(2) * min(dr, dc)


def move_cost(a: Position, b: Position) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if dr == 1 and dc == 1:
        return math.sqrt(2)
    return 1.0


def neighbors8(grid: np.ndarray, pos: Position) -> List[Position]:
    out: List[Position] = []
    for dr, dc in ACTION_DELTAS:
        nxt = (pos[0] + dr, pos[1] + dc)
        if in_bounds(grid, nxt) and is_free(grid, nxt):
            out.append(nxt)
    return out


def reconstruct_path(came_from: Dict[Position, Position], current: Position) -> List[Position]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_path(grid: np.ndarray, start: Position, goal: Position) -> Optional[List[Position]]:
    """Return shortest path from start to goal using A*, or None if unreachable."""
    if not (in_bounds(grid, start) and in_bounds(grid, goal)):
        return None
    if not (is_free(grid, start) and is_free(grid, goal)):
        return None

    open_heap: List[Tuple[float, int, Position]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start))
    tie = 1

    came_from: Dict[Position, Position] = {}
    g_score: Dict[Position, float] = {start: 0.0}
    closed: set[Position] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return reconstruct_path(came_from, current)

        closed.add(current)
        current_g = g_score[current]

        for nbr in neighbors8(grid, current):
            if nbr in closed:
                continue
            tentative_g = current_g + move_cost(current, nbr)
            if tentative_g < g_score.get(nbr, float("inf")):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f = tentative_g + heuristic(nbr, goal)
                heapq.heappush(open_heap, (f, tie, nbr))
                tie += 1

    return None


def sample_start_goal(grid: np.ndarray, rng: np.random.Generator, min_l2: float) -> Optional[Tuple[Position, Position]]:
    free_cells = np.argwhere(grid == 0)
    if free_cells.shape[0] < 2:
        return None

    max_tries = 200
    for _ in range(max_tries):
        idx = rng.choice(free_cells.shape[0], size=2, replace=False)
        s = tuple(map(int, free_cells[idx[0]]))
        g = tuple(map(int, free_cells[idx[1]]))
        if s == g:
            continue
        if np.linalg.norm(np.array(s) - np.array(g), ord=2) >= min_l2:
            return s, g
    return None


def extract_local_patch(grid: np.ndarray, center: Position, patch_size: int) -> np.ndarray:
    """Return square occupancy patch around center; out-of-bounds treated as obstacle."""
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    half = patch_size // 2
    patch = np.ones((patch_size, patch_size), dtype=np.uint8)

    r0, c0 = center
    for pr in range(patch_size):
        for pc in range(patch_size):
            gr = r0 + (pr - half)
            gc = c0 + (pc - half)
            if 0 <= gr < grid.shape[0] and 0 <= gc < grid.shape[1]:
                patch[pr, pc] = grid[gr, gc]
    return patch


def relative_goal(agent: Position, goal: Position, grid_shape: Tuple[int, int]) -> np.ndarray:
    """Normalized relative goal displacement in [-1, 1] approximately."""
    h, w = grid_shape
    dr = (goal[0] - agent[0]) / max(h - 1, 1)
    dc = (goal[1] - agent[1]) / max(w - 1, 1)
    return np.array([dr, dc], dtype=np.float32)


def action_id_from_step(curr: Position, nxt: Position) -> int:
    delta = (nxt[0] - curr[0], nxt[1] - curr[1])
    if delta not in DELTA_TO_ACTION_ID:
        raise ValueError(f"Invalid step delta for 8-connected grid: {delta}")
    return DELTA_TO_ACTION_ID[delta]


def remaining_path_cost(path: List[Position], index: int) -> float:
    """Exact cost from path[index] to goal by summing suffix move costs."""
    if index >= len(path) - 1:
        return 0.0
    cost = 0.0
    for i in range(index, len(path) - 1):
        cost += move_cost(path[i], path[i + 1])
    return float(cost)
