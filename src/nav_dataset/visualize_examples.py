from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .gridworld import ACTION_DELTAS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize sample observations from navigation dataset")
    p.add_argument("--dataset", type=str, default="data/navigation_dataset.npz", help="Path to .npz dataset")
    p.add_argument("--output", type=str, default="visualizations", help="Output directory for figures")
    p.add_argument("--num-examples", type=int, default=12, help="Number of samples to visualize")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    p.add_argument(
        "--mode",
        type=str,
        default="patch",
        choices=["patch", "path"],
        help="patch: local patches, path: full grid path trajectories",
    )
    p.add_argument(
        "--episode-data",
        type=str,
        default="data/episode_data.npz",
        help="Path to episode-level npz file (required for mode=path)",
    )
    p.add_argument(
        "--trajectories",
        type=str,
        default="data/expert_trajectories.jsonl",
        help="Path to episode trajectories jsonl (required for mode=path)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Optional split filter",
    )
    return p


def split_to_code(name: str) -> int:
    mapping = {"train": 0, "val": 1, "test": 2}
    return mapping[name]


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "patch":
        data = np.load(args.dataset)
        patches = data["local_patch"]
        rel_goals = data["relative_goal"]
        actions = data["action_id"]
        costs = data["cost_to_goal"]
        splits = data["split"]

        idx = np.arange(patches.shape[0])
        if args.split != "all":
            idx = idx[splits == split_to_code(args.split)]

        if idx.size == 0:
            raise ValueError("No samples available for selected split")

        rng = np.random.default_rng(args.seed)
        n = min(args.num_examples, idx.size)
        chosen = rng.choice(idx, size=n, replace=False)

        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(rows, cols)

        for k in range(rows * cols):
            ax = axes.flat[k]
            if k >= n:
                ax.axis("off")
                continue

            i = int(chosen[k])
            patch = patches[i]
            rg = rel_goals[i]
            action = int(actions[i])
            cost = float(costs[i])
            dr, dc = ACTION_DELTAS[action]

            ax.imshow(patch, cmap="gray_r", vmin=0, vmax=1)
            c = patch.shape[0] // 2
            ax.scatter([c], [c], c="tab:green", s=40, marker="o", label="agent")
            ax.arrow(c, c, dc * 1.5, dr * 1.5, color="tab:red", head_width=0.5, length_includes_head=True)
            ax.set_title(
                f"idx={i} a={action} cost={cost:.2f}\nrel_goal=({rg[0]:.2f},{rg[1]:.2f})",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        out_file = out_dir / "dataset_examples.png"
        fig.savefig(out_file, dpi=160)
        plt.close(fig)

        print(f"Saved visualization: {out_file}")
        print(f"Plotted {n} samples from split='{args.split}'")
        return

    # mode == path
    import json

    ep_data = np.load(args.episode_data)
    grids = ep_data["episode_grid"]
    ep_splits = ep_data["episode_split"]

    trajectories = []
    with open(args.trajectories, "r", encoding="utf-8") as f:
        for line in f:
            trajectories.append(json.loads(line))

    idx = np.arange(grids.shape[0])
    if args.split != "all":
        idx = idx[ep_splits == split_to_code(args.split)]

    if idx.size == 0:
        raise ValueError("No episodes available for selected split")

    rng = np.random.default_rng(args.seed)
    n = min(args.num_examples, idx.size)
    chosen = rng.choice(idx, size=n, replace=False)

    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for k in range(rows * cols):
        ax = axes.flat[k]
        if k >= n:
            ax.axis("off")
            continue

        ep = int(chosen[k])
        grid = grids[ep]
        rec = trajectories[ep]
        path = np.asarray(rec["path"], dtype=np.int64)
        start = np.asarray(rec["start"], dtype=np.int64)
        goal = np.asarray(rec["goal"], dtype=np.int64)

        ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        ax.plot(path[:, 1], path[:, 0], color="tab:blue", linewidth=2)
        ax.scatter([start[1]], [start[0]], c="tab:green", s=60, marker="o")
        ax.scatter([goal[1]], [goal[0]], c="tab:red", s=60, marker="*")
        ax.set_title(f"episode={ep} len={len(path)} split={int(ep_splits[ep])}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_file = out_dir / "path_examples.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)

    print(f"Saved visualization: {out_file}")
    print(f"Plotted {n} full grid trajectories from split='{args.split}'")


if __name__ == "__main__":
    main()
