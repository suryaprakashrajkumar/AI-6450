import numpy as np

from src.environment import AStarPlanner, Point, path_to_actions


def test_astar_finds_path_on_empty_grid():
    grid = np.zeros((10, 10), dtype=np.int8)
    planner = AStarPlanner()
    path = planner.solve(grid, Point(0, 0), Point(9, 9))
    assert path is not None
    assert path[0] == Point(0, 0)
    assert path[-1] == Point(9, 9)


def test_path_to_actions_non_empty():
    path = [Point(0, 0), Point(1, 1), Point(2, 2)]
    actions = path_to_actions(path)
    assert len(actions) == 2
