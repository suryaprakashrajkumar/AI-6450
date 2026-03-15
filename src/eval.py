from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.environment import AStarPlanner, Point, action_to_delta
from src.model import build_model


def rollout_policy(
    model: torch.nn.Module,
    planner: AStarPlanner,
    grid: np.ndarray,
    start: Point,
    goal: Point,
    step_cap: int,
    enable_astar_fallback: bool,
    enable_confidence_fallback: bool,
    fallback_confidence_threshold: float,
    fallback_min_step: int,
    fallback_no_progress_patience: int,
    fallback_max_calls: int,
) -> tuple[list[Point], bool]:
    model.eval()
    device = next(model.parameters()).device
    grid_t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    path = [start]
    cur = start
    visited: set[tuple[int, int]] = {(cur.row, cur.col)}
    fallback_calls = 0
    no_progress_steps = 0

    def chebyshev_distance(a: Point, b: Point) -> int:
        return max(abs(a.row - b.row), abs(a.col - b.col))

    with torch.no_grad():
        for step_idx in range(step_cap):
            if cur == goal:
                return path, True

            dist_before = chebyshev_distance(cur, goal)

            pos_t = torch.tensor([[cur.row / 9.0, cur.col / 9.0]], dtype=torch.float32, device=device)
            goal_t = torch.tensor([[goal.row / 9.0, goal.col / 9.0]], dtype=torch.float32, device=device)
            logits = model(grid_t, pos_t, goal_t)
            probs = torch.softmax(logits, dim=1)
            max_conf = float(torch.max(probs).item())

            confidence_fallback_ready = (
                enable_astar_fallback
                and enable_confidence_fallback
                and fallback_calls < fallback_max_calls
                and step_idx >= fallback_min_step
                and no_progress_steps >= fallback_no_progress_patience
                and max_conf < fallback_confidence_threshold
            )
            if confidence_fallback_ready:
                fallback_path = planner.solve(grid, cur, goal)
                if fallback_path is not None and len(fallback_path) > 1:
                    fallback_calls += 1
                    path.extend(fallback_path[1:])
                    return path, True

            ranked_actions = torch.argsort(logits, descending=True).squeeze(0).tolist()

            moved = False
            for action in ranked_actions:
                dr, dc = action_to_delta(int(action))
                nr, nc = cur.row + dr, cur.col + dc
                if not (0 <= nr < 10 and 0 <= nc < 10):
                    continue
                if grid[nr, nc] == 1:
                    continue
                nxt = Point(nr, nc)
                if (nr, nc) in visited and nxt != goal:
                    continue
                path.append(nxt)
                cur = nxt
                visited.add((nr, nc))
                moved = True

                dist_after = chebyshev_distance(cur, goal)
                if dist_after < dist_before:
                    no_progress_steps = 0
                else:
                    no_progress_steps += 1
                break

            if not moved:
                if enable_astar_fallback and fallback_calls < fallback_max_calls:
                    fallback_path = planner.solve(grid, cur, goal)
                    if fallback_path is not None and len(fallback_path) > 1:
                        fallback_calls += 1
                        path.extend(fallback_path[1:])
                        return path, True
                return path, False

    return path, cur == goal


def _step_accuracy(model: torch.nn.Module, npz_path: str) -> float:
    payload = np.load(npz_path)
    grids = payload["grids"].astype(np.float32)
    positions = payload["positions"].astype(np.float32)
    goals = payload["goals"].astype(np.float32)
    actions = payload["actions"].astype(np.int64)

    device = next(model.parameters()).device
    correct = 0
    total = len(actions)
    with torch.no_grad():
        for i in range(total):
            grid_t = torch.from_numpy(grids[i]).unsqueeze(0).unsqueeze(0).to(device)
            pos_t = torch.tensor([[positions[i][0] / 9.0, positions[i][1] / 9.0]], dtype=torch.float32, device=device)
            goal_t = torch.tensor([[goals[i][0] / 9.0, goals[i][1] / 9.0]], dtype=torch.float32, device=device)
            pred = model(grid_t, pos_t, goal_t).argmax(dim=1).item()
            if pred == int(actions[i]):
                correct += 1
    return correct / max(total, 1)


def _step_accuracy_limited(model: torch.nn.Module, npz_path: str, limit: int | None = None) -> float:
    payload = np.load(npz_path)
    grids = payload["grids"].astype(np.float32)
    positions = payload["positions"].astype(np.float32)
    goals = payload["goals"].astype(np.float32)
    actions = payload["actions"].astype(np.int64)

    if limit is not None:
        use_n = min(int(limit), len(actions))
        grids = grids[:use_n]
        positions = positions[:use_n]
        goals = goals[:use_n]
        actions = actions[:use_n]

    device = next(model.parameters()).device
    correct = 0
    total = len(actions)
    with torch.no_grad():
        for i in range(total):
            grid_t = torch.from_numpy(grids[i]).unsqueeze(0).unsqueeze(0).to(device)
            pos_t = torch.tensor([[positions[i][0] / 9.0, positions[i][1] / 9.0]], dtype=torch.float32, device=device)
            goal_t = torch.tensor([[goals[i][0] / 9.0, goals[i][1] / 9.0]], dtype=torch.float32, device=device)
            pred = model(grid_t, pos_t, goal_t).argmax(dim=1).item()
            if pred == int(actions[i]):
                correct += 1
    return correct / max(total, 1)


def evaluate_model(
    rollout_npz_path: str,
    model_path: str,
    config_path: str | None = None,
    step_npz_path: str | None = None,
    metrics_output: str | None = None,
    compute_step_accuracy: bool = True,
    rollout_limit: int | None = None,
    step_limit: int | None = None,
) -> dict[str, float | str]:
    config = load_config(config_path)
    payload = np.load(rollout_npz_path)
    grids = payload["grids"]
    starts = payload["starts"]
    goals = payload["goals_full"]

    if rollout_limit is not None:
        use_n = min(int(rollout_limit), len(grids))
        grids = grids[:use_n]
        starts = starts[:use_n]
        goals = goals[:use_n]

    ckpt = torch.load(model_path, map_location="cpu")
    model_type = ckpt.get("model_type", config.model_type)
    model = build_model(
        model_type=model_type,
        hidden_dim=ckpt.get("hidden_dim", config.hidden_dim),
        n_actions=ckpt.get("n_actions", 8),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    planner = AStarPlanner()

    total = len(grids)
    success = 0
    exact = 0
    optimality_gaps: list[float] = []
    valid_path_ratios: list[float] = []

    t_astar = 0.0
    t_policy = 0.0

    for i in range(total):
        grid = grids[i]
        start = Point(int(starts[i][0]), int(starts[i][1]))
        goal = Point(int(goals[i][0]), int(goals[i][1]))

        a0 = time.perf_counter()
        expert_path = planner.solve(grid, start, goal)
        t_astar += time.perf_counter() - a0

        p0 = time.perf_counter()
        pred_path, ok = rollout_policy(
            model,
            planner,
            grid,
            start,
            goal,
            step_cap=config.rollout_step_cap,
            enable_astar_fallback=config.enable_astar_fallback,
            enable_confidence_fallback=config.enable_confidence_fallback,
            fallback_confidence_threshold=config.fallback_confidence_threshold,
            fallback_min_step=config.fallback_min_step,
            fallback_no_progress_patience=config.fallback_no_progress_patience,
            fallback_max_calls=config.fallback_max_calls,
        )
        t_policy += time.perf_counter() - p0

        if expert_path is None:
            continue

        if ok:
            success += 1

        if ok and len(pred_path) == len(expert_path):
            if all((a.row == b.row and a.col == b.col) for a, b in zip(pred_path, expert_path)):
                exact += 1

        if ok:
            gap = (len(pred_path) - len(expert_path)) / max(len(expert_path), 1)
            optimality_gaps.append(gap)
            valid_path_ratios.append(len(pred_path) / max(len(expert_path), 1))

    success_rate = success / max(total, 1)
    exact_rate = exact / max(total, 1)
    mean_gap = float(np.mean(optimality_gaps)) if optimality_gaps else float("inf")
    p95_gap = float(np.percentile(optimality_gaps, 95)) if optimality_gaps else float("inf")
    success_under_1_5x = float(np.mean([g <= 0.5 for g in optimality_gaps])) if optimality_gaps else 0.0
    mean_path_ratio = float(np.mean(valid_path_ratios)) if valid_path_ratios else float("inf")

    astar_ms = (t_astar / max(total, 1)) * 1000.0
    policy_ms = (t_policy / max(total, 1)) * 1000.0
    speedup_astar_over_policy = (astar_ms / policy_ms) if policy_ms > 0 else 0.0
    speedup_policy_over_astar = (policy_ms / astar_ms) if astar_ms > 0 else 0.0

    step_acc = (
        _step_accuracy_limited(model, step_npz_path, limit=step_limit)
        if (compute_step_accuracy and step_npz_path is not None)
        else float("nan")
    )

    metrics: dict[str, float | str] = {
        "model_type": str(model_type),
        "samples": float(total),
        "path_validity": float(success_rate),
        "exact_match": float(exact_rate),
        "step_accuracy": float(step_acc),
        "optimality_gap_mean": float(mean_gap),
        "optimality_gap_p95": float(p95_gap),
        "success_under_1_5x": float(success_under_1_5x),
        "path_length_ratio_mean": float(mean_path_ratio),
        "astar_ms_per_query": float(astar_ms),
        "policy_ms_per_query": float(policy_ms),
        "speedup_astar_over_policy": float(speedup_astar_over_policy),
        "speedup_policy_over_astar": float(speedup_policy_over_astar),
    }

    if metrics_output is not None:
        out = Path(metrics_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=== Evaluation ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")

    return metrics


def evaluate(npz_path: str, model_path: str, config_path: str | None = None) -> None:
    evaluate_model(npz_path, model_path, config_path=config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="data/processed/test_rollout.npz")
    parser.add_argument("--test_steps", type=str, default="data/processed/test_samples.npz")
    parser.add_argument("--model", type=str, default="models/final/imitation_policy.pth")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--metrics_output", type=str, default=None)
    parser.add_argument("--skip_step_accuracy", action="store_true")
    parser.add_argument("--rollout_limit", type=int, default=None)
    parser.add_argument("--step_limit", type=int, default=None)
    args = parser.parse_args()
    evaluate_model(
        rollout_npz_path=args.test,
        step_npz_path=args.test_steps,
        model_path=args.model,
        config_path=args.config,
        metrics_output=args.metrics_output,
        compute_step_accuracy=not args.skip_step_accuracy,
        rollout_limit=args.rollout_limit,
        step_limit=args.step_limit,
    )


if __name__ == "__main__":
    main()
