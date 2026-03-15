from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from src.config import load_config
from src.environment import AStarPlanner, Point, generate_random_grid, path_to_actions, sample_free_point
from src.utils import set_seed


def _split_indices(n_maps: int, train_ratio: float, val_ratio: float):
    idxs = list(range(n_maps))
    random.shuffle(idxs)
    n_train = int(n_maps * train_ratio)
    n_val = int(n_maps * val_ratio)
    train = set(idxs[:n_train])
    val = set(idxs[n_train : n_train + n_val])
    test = set(idxs[n_train + n_val :])
    return train, val, test


def _append_step_samples(grid: np.ndarray, path: list[Point], split_bucket: dict[str, list]):
    actions = path_to_actions(path)
    goal = path[-1]
    for i, action in enumerate(actions):
        cur = path[i]
        split_bucket["grids"].append(grid.copy())
        split_bucket["positions"].append([cur.row, cur.col])
        split_bucket["goals"].append([goal.row, goal.col])
        split_bucket["actions"].append(action)


def _append_rollout_case(grid: np.ndarray, start: Point, goal: Point, split_bucket: dict[str, list]):
    split_bucket["grids"].append(grid.copy())
    split_bucket["starts"].append([start.row, start.col])
    split_bucket["goals_full"].append([goal.row, goal.col])


def _to_npz(path: str, bucket: dict[str, list], keys: list[str]) -> None:
    arrays = {}
    for k in keys:
        arrays[k] = np.array(bucket[k])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _rebalance_actions(bucket: dict[str, list], max_action_ratio: float, rng: random.Random) -> dict[str, list]:
    actions = np.array(bucket["actions"], dtype=np.int64)
    if len(actions) == 0:
        return bucket

    per_action_indices = {a: np.where(actions == a)[0].tolist() for a in range(8)}
    non_zero_counts = [len(v) for v in per_action_indices.values() if len(v) > 0]
    if not non_zero_counts:
        return bucket

    min_count = min(non_zero_counts)
    cap = max(int(min_count * max_action_ratio), min_count)

    keep_indices: list[int] = []
    for action in range(8):
        idxs = per_action_indices[action]
        if not idxs:
            continue
        if len(idxs) > cap:
            sampled = rng.sample(idxs, cap)
            keep_indices.extend(sampled)
        else:
            keep_indices.extend(idxs)

    keep_indices.sort()
    rebalanced = {
        "grids": [bucket["grids"][i] for i in keep_indices],
        "positions": [bucket["positions"][i] for i in keep_indices],
        "goals": [bucket["goals"][i] for i in keep_indices],
        "actions": [bucket["actions"][i] for i in keep_indices],
    }
    return rebalanced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    rng = random.Random(config.seed)

    train_maps, val_maps, test_maps = _split_indices(config.n_maps, config.train_split, config.val_split)
    density_schedule = np.linspace(config.min_obstacle_density, config.max_obstacle_density, num=max(config.density_bins, 2))

    planner = AStarPlanner()

    step_buckets = {
        "train": {"grids": [], "positions": [], "goals": [], "actions": []},
        "val": {"grids": [], "positions": [], "goals": [], "actions": []},
        "test": {"grids": [], "positions": [], "goals": [], "actions": []},
    }

    rollout_buckets = {
        "train": {"grids": [], "starts": [], "goals_full": []},
        "val": {"grids": [], "starts": [], "goals_full": []},
        "test": {"grids": [], "starts": [], "goals_full": []},
    }

    generated_pairs = 0
    kept_pairs = 0

    for map_idx in range(config.n_maps):
        density = float(density_schedule[map_idx % len(density_schedule)])
        grid = generate_random_grid(
            rng,
            config.min_obstacle_density,
            config.max_obstacle_density,
            config.grid_size,
            density_override=density,
        )

        if map_idx in train_maps:
            split = "train"
        elif map_idx in val_maps:
            split = "val"
        else:
            split = "test"

        for _ in range(config.pairs_per_map):
            generated_pairs += 1
            try:
                start = sample_free_point(rng, grid)
                goal = sample_free_point(rng, grid)
            except ValueError:
                continue

            if start == goal:
                continue

            path = planner.solve(grid, start, goal)
            if path is None:
                continue
            if len(path) - 1 < config.min_astar_path_len:
                continue

            _append_step_samples(grid, path, step_buckets[split])
            _append_rollout_case(grid, start, goal, rollout_buckets[split])
            kept_pairs += 1

    if config.rebalance_actions:
        step_buckets["train"] = _rebalance_actions(step_buckets["train"], config.max_action_ratio, rng)

    output_dir = Path(args.output_dir)
    _to_npz(str(output_dir / "train_samples.npz"), step_buckets["train"], ["grids", "positions", "goals", "actions"])
    _to_npz(str(output_dir / "val_samples.npz"), step_buckets["val"], ["grids", "positions", "goals", "actions"])
    _to_npz(str(output_dir / "test_samples.npz"), step_buckets["test"], ["grids", "positions", "goals", "actions"])

    _to_npz(str(output_dir / "train_rollout.npz"), rollout_buckets["train"], ["grids", "starts", "goals_full"])
    _to_npz(str(output_dir / "val_rollout.npz"), rollout_buckets["val"], ["grids", "starts", "goals_full"])
    _to_npz(str(output_dir / "test_rollout.npz"), rollout_buckets["test"], ["grids", "starts", "goals_full"])

    metadata = {
        "seed": config.seed,
        "grid_size": config.grid_size,
        "n_maps": config.n_maps,
        "pairs_per_map": config.pairs_per_map,
        "generated_pairs": generated_pairs,
        "kept_pairs": kept_pairs,
        "min_obstacle_density": config.min_obstacle_density,
        "max_obstacle_density": config.max_obstacle_density,
        "density_bins": config.density_bins,
        "rebalance_actions": config.rebalance_actions,
        "max_action_ratio": config.max_action_ratio,
        "label_source": "A*",
        "temporal_drift_risk": "not_applicable_static_synthetic",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
