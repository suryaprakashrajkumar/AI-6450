from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np

from src.environment import AStarPlanner, Point, path_to_actions
from src.eval import evaluate_model
from src.train import train_model


def _shift_grid_density(grid: np.ndarray, target: float, rng: np.random.Generator, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    out = grid.copy()
    cur = float(out.mean())
    if cur < target:
        add_ratio = min(1.0, (target - cur) / max(1e-6, 1.0 - cur))
        free = out == 0
        mask = (rng.random((10, 10)) < add_ratio) & free
        out[mask] = 1
    out[int(start[0]), int(start[1])] = 0
    out[int(goal[0]), int(goal[1])] = 0
    return out


def _build_shifted_rollout(in_path: str, out_path: str, target_density: float, seed: int) -> None:
    payload = np.load(in_path)
    grids = payload["grids"]
    starts = payload["starts"]
    goals = payload["goals_full"]

    rng = np.random.default_rng(seed)
    out_grids = np.zeros_like(grids)
    for i in range(len(grids)):
        out_grids[i] = _shift_grid_density(grids[i], target_density, rng, starts[i], goals[i])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, grids=out_grids, starts=starts, goals_full=goals)


def _build_augmented_steps(
    train_steps: str,
    train_rollout: str,
    out_path: str,
    ratio: float,
    target_density: float,
    seed: int,
) -> None:
    base = np.load(train_steps)
    base_grids = base["grids"]
    base_positions = base["positions"]
    base_goals = base["goals"]
    base_actions = base["actions"]

    rollout = np.load(train_rollout)
    r_grids = rollout["grids"]
    r_starts = rollout["starts"]
    r_goals = rollout["goals_full"]

    planner = AStarPlanner()
    rng = np.random.default_rng(seed)

    target_new = max(1, int(len(base_actions) * ratio))
    case_order = rng.permutation(len(r_grids))

    add_grids: list[np.ndarray] = []
    add_positions: list[list[int]] = []
    add_goals: list[list[int]] = []
    add_actions: list[int] = []

    for ci in case_order:
        if len(add_actions) >= target_new:
            break

        grid = r_grids[ci]
        st = r_starts[ci]
        gl = r_goals[ci]
        shifted = _shift_grid_density(grid, target_density, rng, st, gl)

        start = Point(int(st[0]), int(st[1]))
        goal = Point(int(gl[0]), int(gl[1]))
        path = planner.solve(shifted, start, goal)
        if path is None or len(path) < 2:
            continue

        acts = path_to_actions(path)
        for i, a in enumerate(acts):
            cur = path[i]
            add_grids.append(shifted.copy())
            add_positions.append([cur.row, cur.col])
            add_goals.append([goal.row, goal.col])
            add_actions.append(int(a))
            if len(add_actions) >= target_new:
                break

    if not add_actions:
        raise RuntimeError("Could not build any valid shifted expert-labeled samples for adaptation.")

    aug_grids = np.concatenate([base_grids, np.array(add_grids)], axis=0)
    aug_positions = np.concatenate([base_positions, np.array(add_positions)], axis=0)
    aug_goals = np.concatenate([base_goals, np.array(add_goals)], axis=0)
    aug_actions = np.concatenate([base_actions, np.array(add_actions)], axis=0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        grids=aug_grids,
        positions=aug_positions,
        goals=aug_goals,
        actions=aug_actions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="config_improved.json")
    parser.add_argument("--base_model", type=str, default="models/final/imitation_policy_improved.pth")
    parser.add_argument("--train_steps", type=str, default="data/processed/train_samples.npz")
    parser.add_argument("--train_rollout", type=str, default="data/processed/train_rollout.npz")
    parser.add_argument("--val_steps", type=str, default="data/processed/val_samples.npz")
    parser.add_argument("--test_steps", type=str, default="data/processed/test_samples.npz")
    parser.add_argument("--test_rollout", type=str, default="data/processed/test_rollout.npz")
    parser.add_argument("--out_dir", type=str, default="logs/adaptation")
    parser.add_argument("--shift_target_density", type=float, default=0.50)
    parser.add_argument("--augment_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shifted_rollout = out_dir / "test_rollout_shifted.npz"
    _build_shifted_rollout(args.test_rollout, str(shifted_rollout), args.shift_target_density, args.seed)

    # Before adaptation
    before = evaluate_model(
        rollout_npz_path=str(shifted_rollout),
        step_npz_path=args.test_steps,
        model_path=args.base_model,
        config_path=args.base_config,
        metrics_output=str(out_dir / "before_metrics.json"),
        compute_step_accuracy=False,
        rollout_limit=400,
    )

    # Prepare augmented train set for adaptation
    augmented_steps = out_dir / "train_augmented_steps.npz"
    _build_augmented_steps(
        train_steps=args.train_steps,
        train_rollout=args.train_rollout,
        out_path=str(augmented_steps),
        ratio=args.augment_ratio,
        target_density=args.shift_target_density,
        seed=args.seed,
    )

    # Create temporary config with shorter adaptation training schedule
    cfg = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    cfg["epochs"] = int(args.epochs)
    cfg["early_stop_patience"] = min(4, int(args.epochs))
    cfg["lr"] = 6e-4
    temp_cfg = out_dir / "adapt_config.json"
    temp_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    adapted_model = out_dir / "adapted_model.pth"
    train_model(
        config_path=str(temp_cfg),
        train_npz=str(augmented_steps),
        val_npz=args.val_steps,
        output_path=str(adapted_model),
        metrics_output=str(out_dir / "adapt_train_metrics.json"),
        history_output=str(out_dir / "adapt_train_history.csv"),
        history_plot_output=str(out_dir / "adapt_train_curves.png"),
    )

    # After adaptation
    after = evaluate_model(
        rollout_npz_path=str(shifted_rollout),
        step_npz_path=args.test_steps,
        model_path=str(adapted_model),
        config_path=str(temp_cfg),
        metrics_output=str(out_dir / "after_metrics.json"),
        compute_step_accuracy=False,
        rollout_limit=400,
    )

    rows = [
        {
            "stage": "before",
            "path_validity": float(before["path_validity"]),
            "exact_match": float(before["exact_match"]),
            "optimality_gap_mean": float(before["optimality_gap_mean"]),
            "policy_ms_per_query": float(before["policy_ms_per_query"]),
        },
        {
            "stage": "after",
            "path_validity": float(after["path_validity"]),
            "exact_match": float(after["exact_match"]),
            "optimality_gap_mean": float(after["optimality_gap_mean"]),
            "policy_ms_per_query": float(after["policy_ms_per_query"]),
        },
    ]

    with (out_dir / "before_after_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "path_validity", "exact_match", "optimality_gap_mean", "policy_ms_per_query"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Plot before/after
    labels = ["path_validity", "exact_match", "optimality_gap_mean"]
    x = np.arange(len(labels))
    before_vals = [rows[0][k] for k in labels]
    after_vals = [rows[1][k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.35
    ax.bar(x - w / 2, before_vals, width=w, label="before")
    ax.bar(x + w / 2, after_vals, width=w, label="after")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Adaptation Before/After on Shifted Evaluation")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "before_after_comparison.png", dpi=170)
    plt.close(fig)

    summary = {
        "shift_target_density": args.shift_target_density,
        "augment_ratio": args.augment_ratio,
        "rollout_limit": 400,
        "before_path_validity": float(before["path_validity"]),
        "after_path_validity": float(after["path_validity"]),
        "before_exact_match": float(before["exact_match"]),
        "after_exact_match": float(after["exact_match"]),
    }
    (out_dir / "adaptation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
