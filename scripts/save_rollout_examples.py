from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import torch

from src.config import load_config
from src.environment import AStarPlanner, Point
from src.eval import rollout_policy
from src.model import build_model


def _path_to_xy(path: list[Point]) -> tuple[list[int], list[int]]:
    xs = [p.col for p in path]
    ys = [p.row for p in path]
    return xs, ys


def _plot_case(
    grid: np.ndarray,
    expert_path: list[Point] | None,
    pred_path: list[Point],
    success: bool,
    case_id: int,
    gap: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))

    # Obstacles in black, free space in white.
    ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)

    if expert_path is not None and len(expert_path) > 0:
        ex_x, ex_y = _path_to_xy(expert_path)
        ax.plot(ex_x, ex_y, color="#1D4ED8", linewidth=2.5, label="A* expert")

    if len(pred_path) > 0:
        pr_x, pr_y = _path_to_xy(pred_path)
        ax.plot(pr_x, pr_y, color="#D97706", linewidth=2.0, linestyle="--", label="Policy")

        start = pred_path[0]
        end = pred_path[-1]
        ax.scatter([start.col], [start.row], c="#059669", s=70, marker="o", label="Start")
        ax.scatter([end.col], [end.row], c="#DC2626", s=85, marker="*", label="End")

    ax.set_title(f"Case {case_id} | success={success} | gap={gap:.3f}")
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(9.5, -0.5)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.grid(color="#9CA3AF", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _select_examples(rows: list[dict], n: int) -> list[dict]:
    if not rows:
        return []

    rows_sorted = sorted(rows, key=lambda r: (0 if r["success"] else 1, float(r["gap"])))

    picks: list[dict] = []

    # Always include the best and worst (by gap), then quantile picks.
    picks.append(rows_sorted[0])
    picks.append(rows_sorted[-1])

    if n > 2:
        quantiles = np.linspace(0, len(rows_sorted) - 1, num=n)
        for q in quantiles:
            picks.append(rows_sorted[int(round(float(q)))])

    unique: dict[int, dict] = {}
    for r in picks:
        unique[int(r["index"])] = r

    selected = list(unique.values())
    selected.sort(key=lambda r: float(r["gap"]))
    return selected[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_improved.json")
    parser.add_argument("--model", type=str, default="models/final/imitation_policy_improved.pth")
    parser.add_argument("--test_rollout", type=str, default="data/processed/test_rollout.npz")
    parser.add_argument("--out_dir", type=str, default="logs/rollout_examples")
    parser.add_argument("--num_examples", type=int, default=12)
    parser.add_argument("--scan_limit", type=int, default=240)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.test_rollout)
    grids = payload["grids"]
    starts = payload["starts"]
    goals = payload["goals_full"]

    ckpt = torch.load(args.model, map_location="cpu")
    model = build_model(
        model_type=ckpt.get("model_type", cfg.model_type),
        hidden_dim=ckpt.get("hidden_dim", cfg.hidden_dim),
        n_actions=ckpt.get("n_actions", cfg.n_actions),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    planner = AStarPlanner()

    max_n = min(int(args.scan_limit), len(grids))
    rows: list[dict] = []

    for i in range(max_n):
        grid = grids[i]
        start = Point(int(starts[i][0]), int(starts[i][1]))
        goal = Point(int(goals[i][0]), int(goals[i][1]))

        expert_path = planner.solve(grid, start, goal)
        pred_path, success = rollout_policy(
            model=model,
            planner=planner,
            grid=grid,
            start=start,
            goal=goal,
            step_cap=cfg.rollout_step_cap,
            enable_astar_fallback=cfg.enable_astar_fallback,
            enable_confidence_fallback=cfg.enable_confidence_fallback,
            fallback_confidence_threshold=cfg.fallback_confidence_threshold,
            fallback_min_step=cfg.fallback_min_step,
            fallback_no_progress_patience=cfg.fallback_no_progress_patience,
            fallback_max_calls=cfg.fallback_max_calls,
        )

        if expert_path is None:
            continue

        gap = (len(pred_path) - len(expert_path)) / max(1, len(expert_path)) if success else 999.0

        rows.append(
            {
                "index": i,
                "success": bool(success),
                "expert_len": len(expert_path),
                "pred_len": len(pred_path),
                "gap": float(gap),
            }
        )

    selected = _select_examples(rows, max(1, int(args.num_examples)))

    for row in selected:
        idx = int(row["index"])
        grid = grids[idx]
        start = Point(int(starts[idx][0]), int(starts[idx][1]))
        goal = Point(int(goals[idx][0]), int(goals[idx][1]))
        expert_path = planner.solve(grid, start, goal)
        pred_path, success = rollout_policy(
            model=model,
            planner=planner,
            grid=grid,
            start=start,
            goal=goal,
            step_cap=cfg.rollout_step_cap,
            enable_astar_fallback=cfg.enable_astar_fallback,
            enable_confidence_fallback=cfg.enable_confidence_fallback,
            fallback_confidence_threshold=cfg.fallback_confidence_threshold,
            fallback_min_step=cfg.fallback_min_step,
            fallback_no_progress_patience=cfg.fallback_no_progress_patience,
            fallback_max_calls=cfg.fallback_max_calls,
        )

        case_path = out_dir / f"case_{idx:04d}.png"
        _plot_case(
            grid=grid,
            expert_path=expert_path,
            pred_path=pred_path,
            success=success,
            case_id=idx,
            gap=float(row["gap"]),
            out_path=case_path,
        )

    csv_path = out_dir / "examples_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "success", "expert_len", "pred_len", "gap"])
        writer.writeheader()
        writer.writerows(selected)

    stats = {
        "scanned_cases": len(rows),
        "selected_examples": len(selected),
        "success_rate_scanned": float(np.mean([1.0 if r["success"] else 0.0 for r in rows])) if rows else 0.0,
        "mean_gap_scanned": float(np.mean([r["gap"] for r in rows])) if rows else float("nan"),
    }
    (out_dir / "examples_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Saved rollout examples to {out_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
