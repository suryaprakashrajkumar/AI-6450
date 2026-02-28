from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .gridworld import (
    GridConfig,
    action_id_from_step,
    astar_path,
    extract_local_patch,
    make_grid,
    relative_goal,
    remaining_path_cost,
    sample_start_goal,
)


@dataclass(frozen=True)
class DatasetConfig:
    episodes: int = 1000
    patch_size: int = 15
    max_episode_sampling_attempts: int = 50
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


def _validate_split(cfg: DatasetConfig) -> None:
    s = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if not np.isclose(s, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {s}")


def _split_code(r: float, cfg: DatasetConfig) -> int:
    if r < cfg.train_ratio:
        return 0
    if r < cfg.train_ratio + cfg.val_ratio:
        return 1
    return 2


def collect_dataset(grid_cfg: GridConfig, data_cfg: DatasetConfig, seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Build supervised+imitation dataset from A* demonstrations.

    Returns arrays keyed by field names.
    """
    _validate_split(data_cfg)
    rng = np.random.default_rng(seed)

    local_patch_list: List[np.ndarray] = []
    rel_goal_list: List[np.ndarray] = []
    action_id_list: List[int] = []
    cost_to_goal_list: List[float] = []
    agent_position_list: List[List[int]] = []
    episode_id_list: List[int] = []
    timestep_list: List[int] = []
    split_list: List[int] = []

    episode_grid_list: List[np.ndarray] = []
    episode_start_list: List[List[int]] = []
    episode_goal_list: List[List[int]] = []
    episode_split_list: List[int] = []
    episode_path_list: List[List[List[int]]] = []

    accepted = 0
    attempts = 0
    max_total_attempts = data_cfg.episodes * data_cfg.max_episode_sampling_attempts

    while accepted < data_cfg.episodes and attempts < max_total_attempts:
        attempts += 1
        grid = make_grid(grid_cfg, rng)
        start_goal = sample_start_goal(grid, rng, min_l2=grid_cfg.min_start_goal_l2)
        if start_goal is None:
            continue

        start, goal = start_goal
        path = astar_path(grid, start, goal)
        if path is None or len(path) < 2:
            continue

        split_code = _split_code(float(rng.random()), data_cfg)

        for t in range(len(path) - 1):
            agent = path[t]
            nxt = path[t + 1]

            local_patch_list.append(extract_local_patch(grid, agent, data_cfg.patch_size))
            rel_goal_list.append(relative_goal(agent, goal, grid.shape))
            action_id_list.append(action_id_from_step(agent, nxt))
            cost_to_goal_list.append(remaining_path_cost(path, t))
            agent_position_list.append([int(agent[0]), int(agent[1])])
            episode_id_list.append(accepted)
            timestep_list.append(t)
            split_list.append(split_code)

        episode_grid_list.append(grid.copy())
        episode_start_list.append([int(start[0]), int(start[1])])
        episode_goal_list.append([int(goal[0]), int(goal[1])])
        episode_split_list.append(split_code)
        episode_path_list.append([[int(p[0]), int(p[1])] for p in path])

        accepted += 1

    if accepted < data_cfg.episodes:
        raise RuntimeError(
            f"Could only collect {accepted}/{data_cfg.episodes} episodes after {attempts} attempts. "
            "Try lowering obstacle density or start-goal distance threshold."
        )

    data = {
        "local_patch": np.stack(local_patch_list).astype(np.uint8),
        "relative_goal": np.stack(rel_goal_list).astype(np.float32),
        "action_id": np.asarray(action_id_list, dtype=np.int64),
        "cost_to_goal": np.asarray(cost_to_goal_list, dtype=np.float32),
        "agent_position": np.asarray(agent_position_list, dtype=np.int64),
        "episode_id": np.asarray(episode_id_list, dtype=np.int64),
        "timestep": np.asarray(timestep_list, dtype=np.int64),
        "split": np.asarray(split_list, dtype=np.int8),
        "episode_grid": np.stack(episode_grid_list).astype(np.uint8),
        "episode_start": np.asarray(episode_start_list, dtype=np.int64),
        "episode_goal": np.asarray(episode_goal_list, dtype=np.int64),
        "episode_split": np.asarray(episode_split_list, dtype=np.int8),
        "episode_path": np.asarray(episode_path_list, dtype=object),
    }
    return data


def save_dataset(
    output_dir: str | Path,
    data: Dict[str, np.ndarray],
    grid_cfg: GridConfig,
    data_cfg: DatasetConfig,
    seed: int,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    npz_path = output / "navigation_dataset.npz"
    # Save per-step supervised arrays in the primary file.
    np.savez_compressed(
        npz_path,
        local_patch=data["local_patch"],
        relative_goal=data["relative_goal"],
        action_id=data["action_id"],
        cost_to_goal=data["cost_to_goal"],
        agent_position=data["agent_position"],
        episode_id=data["episode_id"],
        timestep=data["timestep"],
        split=data["split"],
    )

    # Save episode-level artifacts (complete grid paths and full occupancy maps).
    episode_npz_path = output / "episode_data.npz"
    np.savez_compressed(
        episode_npz_path,
        episode_grid=data["episode_grid"],
        episode_start=data["episode_start"],
        episode_goal=data["episode_goal"],
        episode_split=data["episode_split"],
    )

    trajectories_path = output / "expert_trajectories.jsonl"
    with trajectories_path.open("w", encoding="utf-8") as f:
        for ep in range(int(data["episode_start"].shape[0])):
            rec = {
                "episode_id": ep,
                "split": int(data["episode_split"][ep]),
                "start": data["episode_start"][ep].tolist(),
                "goal": data["episode_goal"][ep].tolist(),
                "path": data["episode_path"][ep],
            }
            f.write(json.dumps(rec) + "\n")

    split = data["split"]
    split_counts = {
        "train": int(np.sum(split == 0)),
        "val": int(np.sum(split == 1)),
        "test": int(np.sum(split == 2)),
    }

    metadata = {
        "seed": seed,
        "grid_config": asdict(grid_cfg),
        "dataset_config": asdict(data_cfg),
        "num_samples": int(data["action_id"].shape[0]),
        "num_episodes": int(np.unique(data["episode_id"]).shape[0]),
        "split_counts": split_counts,
        "fields": {
            "local_patch": "(N,P,P) uint8 obstacle map patch",
            "relative_goal": "(N,2) float32 normalized [dr, dc]",
            "action_id": "(N,) int64 in [0..7]",
            "cost_to_goal": "(N,) float32",
            "agent_position": "(N,2) int64 [row, col]",
            "episode_id": "(N,) int64",
            "timestep": "(N,) int64",
            "split": "(N,) int8 0=train,1=val,2=test",
        },
        "episode_artifacts": {
            "episode_data.npz": {
                "episode_grid": "(E,H,W) uint8 full occupancy grid per episode",
                "episode_start": "(E,2) int64",
                "episode_goal": "(E,2) int64",
                "episode_split": "(E,) int8 0=train,1=val,2=test",
            },
            "expert_trajectories.jsonl": "One record per episode with complete grid-based path coordinates",
        },
    }

    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
