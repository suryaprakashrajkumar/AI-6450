from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
import math
import random
from typing import Iterable

import numpy as np


# 8-direction movement: N, NE, E, SE, S, SW, W, NW
ACTIONS: list[tuple[int, int]] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]


def action_to_delta(action_id: int) -> tuple[int, int]:
    return ACTIONS[action_id]


def delta_to_action(delta: tuple[int, int]) -> int:
    return ACTIONS.index(delta)


@dataclass(frozen=True)
class Point:
    row: int
    col: int


class GridWorld:
    def __init__(self, grid: np.ndarray):
        if grid.shape != (10, 10):
            raise ValueError("Grid must be 10x10.")
        self.grid = grid.astype(np.int8)

    def in_bounds(self, point: Point) -> bool:
        return 0 <= point.row < 10 and 0 <= point.col < 10

    def is_free(self, point: Point) -> bool:
        return self.in_bounds(point) and self.grid[point.row, point.col] == 0

    def neighbors(self, point: Point) -> Iterable[Point]:
        for dr, dc in ACTIONS:
            nxt = Point(point.row + dr, point.col + dc)
            if self.is_free(nxt):
                yield nxt


class AStarPlanner:
    def __init__(self) -> None:
        self._tie = 0

    @staticmethod
    def heuristic(a: Point, b: Point) -> float:
        return math.hypot(a.row - b.row, a.col - b.col)

    @staticmethod
    def distance(_: Point, __: Point) -> float:
        return 1.0

    def solve(self, grid: np.ndarray, start: Point, goal: Point) -> list[Point] | None:
        world = GridWorld(grid)
        if not world.is_free(start) or not world.is_free(goal):
            return None

        open_heap: list[tuple[float, int, Point]] = []
        came_from: dict[Point, Point | None] = {start: None}
        g_score: dict[Point, float] = {start: 0.0}

        self._tie += 1
        heappush(open_heap, (self.heuristic(start, goal), self._tie, start))

        while open_heap:
            _, _, current = heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for nxt in world.neighbors(current):
                tentative = g_score[current] + self.distance(current, nxt)
                if tentative < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    f_score = tentative + self.heuristic(nxt, goal)
                    self._tie += 1
                    heappush(open_heap, (f_score, self._tie, nxt))

        return None

    @staticmethod
    def _reconstruct_path(came_from: dict[Point, Point | None], current: Point) -> list[Point]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]  # type: ignore[assignment]
            path.append(current)
        path.reverse()
        return path


def generate_random_grid(
    rng: random.Random,
    min_density: float,
    max_density: float,
    grid_size: int = 10,
    density_override: float | None = None,
) -> np.ndarray:
    density = density_override if density_override is not None else rng.uniform(min_density, max_density)
    grid_arr = np.zeros((grid_size, grid_size), dtype=np.int8)
    for r in range(grid_size):
        for c in range(grid_size):
            grid_arr[r, c] = 1 if (rng.random() < density) else 0
    return grid_arr


def sample_free_point(rng: random.Random, grid: np.ndarray) -> Point:
    free = np.argwhere(grid == 0)
    if free.size == 0:
        raise ValueError("Grid has no free cells.")
    idx = rng.randrange(len(free))
    row, col = free[idx]
    return Point(int(row), int(col))


def path_to_actions(path: list[Point]) -> list[int]:
    actions: list[int] = []
    for i in range(len(path) - 1):
        delta = (path[i + 1].row - path[i].row, path[i + 1].col - path[i].col)
        actions.append(delta_to_action(delta))
    return actions


def is_valid_transition(grid: np.ndarray, cur: Point, action_id: int) -> bool:
    dr, dc = action_to_delta(action_id)
    nxt = Point(cur.row + dr, cur.col + dc)
    return 0 <= nxt.row < 10 and 0 <= nxt.col < 10 and grid[nxt.row, nxt.col] == 0
