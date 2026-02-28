from __future__ import annotations

import argparse
import time

from .collector import DatasetConfig, collect_dataset, save_dataset
from .gridworld import GridConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate dataset for 2-D grid navigation learning")
    p.add_argument("--output", type=str, default="data", help="Output directory")
    p.add_argument("--episodes", type=int, default=1000, help="Number of successful expert episodes")
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--obstacle-density", type=float, default=0.2)
    p.add_argument("--min-start-goal-l2", type=float, default=12.0)
    p.add_argument(
        "--map-type",
        type=str,
        default="random",
        choices=["random", "maze"],
        help="Map generator type",
    )
    p.add_argument(
        "--maze-extra-open-prob",
        type=float,
        default=0.02,
        help="Extra random wall openings for maze maps (adds loops)",
    )
    p.add_argument("--patch-size", type=int, default=15)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()

    grid_cfg = GridConfig(
        height=args.height,
        width=args.width,
        obstacle_density=args.obstacle_density,
        min_start_goal_l2=args.min_start_goal_l2,
        map_type=args.map_type,
        maze_extra_open_prob=args.maze_extra_open_prob,
    )
    data_cfg = DatasetConfig(
        episodes=args.episodes,
        patch_size=args.patch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    t0 = time.perf_counter()
    data = collect_dataset(grid_cfg=grid_cfg, data_cfg=data_cfg, seed=args.seed)
    save_dataset(args.output, data, grid_cfg, data_cfg, seed=args.seed)
    dt = time.perf_counter() - t0

    print("Dataset generation complete")
    print(f"Output dir: {args.output}")
    print(f"Samples: {data['action_id'].shape[0]}")
    print(f"Episodes: {args.episodes}")
    print(f"Elapsed: {dt:.2f}s")


if __name__ == "__main__":
    main()
