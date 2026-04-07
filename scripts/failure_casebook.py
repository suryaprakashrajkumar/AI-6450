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
from src.utils import get_torch_device


def _gap(pred_len: int, expert_len: int, success: bool) -> float:
    if not success:
        return 9.99
    return (pred_len - expert_len) / max(expert_len, 1)


def _root_cause(grid: np.ndarray, base_success: bool, base_gap: float) -> str:
    density = float(grid.mean())
    if density > 0.45:
        return "high_obstacle_density"
    if not base_success:
        return "search_stall_or_dead_end"
    if base_gap > 0.8:
        return "large_detour"
    return "route_instability"


def _plot_case(grid: np.ndarray, expert: list[Point], base: list[Point], adapted: list[Point], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)

    ex_x = [p.col for p in expert]
    ex_y = [p.row for p in expert]
    ax.plot(ex_x, ex_y, color="#1D4ED8", linewidth=2.4, label="A* expert")

    b_x = [p.col for p in base]
    b_y = [p.row for p in base]
    ax.plot(b_x, b_y, color="#B91C1C", linewidth=2.0, linestyle="--", label="before adapt")

    a_x = [p.col for p in adapted]
    a_y = [p.row for p in adapted]
    ax.plot(a_x, a_y, color="#047857", linewidth=2.0, linestyle="-.", label="after adapt")

    ax.set_title(title)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(9.5, -0.5)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _load_model(model_path: str, config_path: str):
    cfg = load_config(config_path)
    device = get_torch_device()
    ckpt = torch.load(model_path, map_location=device)
    model = build_model(
        model_type=ckpt.get("model_type", cfg.model_type),
        hidden_dim=ckpt.get("hidden_dim", cfg.hidden_dim),
        n_actions=ckpt.get("n_actions", cfg.n_actions),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="logs/phase2/config_phase2_best.json")
    parser.add_argument("--base_model", type=str, default="models/final/imitation_policy_improved.pth")
    parser.add_argument("--adapt_config", type=str, default="logs/adaptation/adapt_config.json")
    parser.add_argument("--adapt_model", type=str, default="logs/adaptation/adapted_model.pth")
    parser.add_argument("--shifted_rollout", type=str, default="logs/adaptation/test_rollout_shifted.npz")
    parser.add_argument("--out_dir", type=str, default="logs/failure_casebook")
    parser.add_argument("--scan_limit", type=int, default=400)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model, base_cfg = _load_model(args.base_model, args.base_config)
    adapted_model, adapted_cfg = _load_model(args.adapt_model, args.adapt_config)

    payload = np.load(args.shifted_rollout)
    grids = payload["grids"]
    starts = payload["starts"]
    goals = payload["goals_full"]

    planner = AStarPlanner()

    rows: list[dict[str, float | int | str]] = []
    n = min(int(args.scan_limit), len(grids))

    for i in range(n):
        grid = grids[i]
        start = Point(int(starts[i][0]), int(starts[i][1]))
        goal = Point(int(goals[i][0]), int(goals[i][1]))

        expert = planner.solve(grid, start, goal)
        if expert is None or len(expert) < 2:
            continue

        base_path, base_ok = rollout_policy(
            model=base_model,
            planner=planner,
            grid=grid,
            start=start,
            goal=goal,
            step_cap=base_cfg.rollout_step_cap,
            enable_astar_fallback=base_cfg.enable_astar_fallback,
            enable_confidence_fallback=base_cfg.enable_confidence_fallback,
            fallback_confidence_threshold=base_cfg.fallback_confidence_threshold,
            fallback_min_step=base_cfg.fallback_min_step,
            fallback_no_progress_patience=base_cfg.fallback_no_progress_patience,
            fallback_max_calls=base_cfg.fallback_max_calls,
        )

        adapt_path, adapt_ok = rollout_policy(
            model=adapted_model,
            planner=planner,
            grid=grid,
            start=start,
            goal=goal,
            step_cap=adapted_cfg.rollout_step_cap,
            enable_astar_fallback=adapted_cfg.enable_astar_fallback,
            enable_confidence_fallback=adapted_cfg.enable_confidence_fallback,
            fallback_confidence_threshold=adapted_cfg.fallback_confidence_threshold,
            fallback_min_step=adapted_cfg.fallback_min_step,
            fallback_no_progress_patience=adapted_cfg.fallback_no_progress_patience,
            fallback_max_calls=adapted_cfg.fallback_max_calls,
        )

        base_gap = _gap(len(base_path), len(expert), base_ok)
        adapt_gap = _gap(len(adapt_path), len(expert), adapt_ok)

        resolved = (not base_ok and adapt_ok) or (adapt_gap + 0.2 < base_gap)
        root = _root_cause(grid, base_ok, base_gap)

        rows.append(
            {
                "index": int(i),
                "density": float(grid.mean()),
                "expert_len": int(len(expert)),
                "base_success": int(1 if base_ok else 0),
                "adapt_success": int(1 if adapt_ok else 0),
                "base_gap": float(base_gap),
                "adapt_gap": float(adapt_gap),
                "delta_gap": float(base_gap - adapt_gap),
                "root_cause": root,
                "resolved": int(1 if resolved else 0),
            }
        )

    # Rank by severity and improvement to keep high-value cases.
    rows_sorted = sorted(rows, key=lambda r: (-(float(r["base_gap"])), -float(r["delta_gap"])))
    selected = rows_sorted[: max(6, min(args.top_k, len(rows_sorted)))]

    # Ensure at least 2 resolved if possible.
    resolved_rows = [r for r in rows_sorted if int(r["resolved"]) == 1]
    if len([r for r in selected if int(r["resolved"]) == 1]) < 2 and len(resolved_rows) >= 2:
        selected[-2:] = resolved_rows[:2]

    selected_unique = {int(r["index"]): r for r in selected}
    selected = list(selected_unique.values())

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for r in selected:
        i = int(r["index"])
        grid = grids[i]
        start = Point(int(starts[i][0]), int(starts[i][1]))
        goal = Point(int(goals[i][0]), int(goals[i][1]))
        expert = planner.solve(grid, start, goal)
        if expert is None:
            continue

        base_path, _ = rollout_policy(
            model=base_model,
            planner=planner,
            grid=grid,
            start=start,
            goal=goal,
            step_cap=base_cfg.rollout_step_cap,
            enable_astar_fallback=base_cfg.enable_astar_fallback,
            enable_confidence_fallback=base_cfg.enable_confidence_fallback,
            fallback_confidence_threshold=base_cfg.fallback_confidence_threshold,
            fallback_min_step=base_cfg.fallback_min_step,
            fallback_no_progress_patience=base_cfg.fallback_no_progress_patience,
            fallback_max_calls=base_cfg.fallback_max_calls,
        )
        adapt_path, _ = rollout_policy(
            model=adapted_model,
            planner=planner,
            grid=grid,
            start=start,
            goal=goal,
            step_cap=adapted_cfg.rollout_step_cap,
            enable_astar_fallback=adapted_cfg.enable_astar_fallback,
            enable_confidence_fallback=adapted_cfg.enable_confidence_fallback,
            fallback_confidence_threshold=adapted_cfg.fallback_confidence_threshold,
            fallback_min_step=adapted_cfg.fallback_min_step,
            fallback_no_progress_patience=adapted_cfg.fallback_no_progress_patience,
            fallback_max_calls=adapted_cfg.fallback_max_calls,
        )

        title = f"case {i} | resolved={int(r['resolved'])} | cause={r['root_cause']}"
        _plot_case(grid, expert, base_path, adapt_path, title, images_dir / f"case_{i:04d}.png")

    csv_path = out_dir / "failure_casebook.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "density",
                "expert_len",
                "base_success",
                "adapt_success",
                "base_gap",
                "adapt_gap",
                "delta_gap",
                "root_cause",
                "resolved",
            ],
        )
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "selected_cases": len(selected),
        "resolved_cases": int(sum(int(r["resolved"]) for r in selected)),
        "csv": str(csv_path),
        "images_dir": str(images_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
