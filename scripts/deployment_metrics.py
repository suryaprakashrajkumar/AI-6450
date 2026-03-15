from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.environment import AStarPlanner, Point
from src.eval import rollout_policy
from src.model import build_model


def _percentiles_ms(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="logs/phase2/config_phase2_best.json")
    parser.add_argument("--model", type=str, default="models/final/imitation_policy_improved.pth")
    parser.add_argument("--test_rollout", type=str, default="data/processed/test_rollout.npz")
    parser.add_argument("--out_dir", type=str, default="logs/deployment_metrics")
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    payload = np.load(args.test_rollout)
    grids = payload["grids"]
    starts = payload["starts"]
    goals = payload["goals_full"]

    n = min(int(args.limit), len(grids))

    ckpt = torch.load(args.model, map_location="cpu")
    model = build_model(
        model_type=ckpt.get("model_type", cfg.model_type),
        hidden_dim=ckpt.get("hidden_dim", cfg.hidden_dim),
        n_actions=ckpt.get("n_actions", cfg.n_actions),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    planner = AStarPlanner()

    astar_ms: list[float] = []
    policy_ms: list[float] = []
    per_case_rows: list[dict[str, float | int | str]] = []

    for i in range(n):
        grid = grids[i]
        start = Point(int(starts[i][0]), int(starts[i][1]))
        goal = Point(int(goals[i][0]), int(goals[i][1]))

        t0 = time.perf_counter()
        expert = planner.solve(grid, start, goal)
        t1 = time.perf_counter()
        a_ms = (t1 - t0) * 1000.0

        t2 = time.perf_counter()
        pred, ok = rollout_policy(
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
        t3 = time.perf_counter()
        p_ms = (t3 - t2) * 1000.0

        astar_ms.append(a_ms)
        policy_ms.append(p_ms)

        per_case_rows.append(
            {
                "index": int(i),
                "astar_ms": float(a_ms),
                "policy_ms": float(p_ms),
                "success": int(1 if ok else 0),
                "expert_len": int(len(expert) if expert is not None else -1),
                "pred_len": int(len(pred)),
            }
        )

    astar_stats = _percentiles_ms(astar_ms)
    policy_stats = _percentiles_ms(policy_ms)

    astar_qps = float(1000.0 / astar_stats["mean"]) if astar_stats["mean"] > 0 else 0.0
    policy_qps = float(1000.0 / policy_stats["mean"]) if policy_stats["mean"] > 0 else 0.0

    file_size_mb = float(Path(args.model).stat().st_size / (1024 * 1024))
    param_count = int(sum(p.numel() for p in model.parameters()))
    param_mem_mb = float(param_count * 4 / (1024 * 1024))

    summary = {
        "samples": int(n),
        "astar_ms": astar_stats,
        "policy_ms": policy_stats,
        "astar_qps": astar_qps,
        "policy_qps": policy_qps,
        "model_file_size_mb": file_size_mb,
        "model_parameter_count": param_count,
        "model_parameter_memory_mb_fp32": param_mem_mb,
    }

    (out_dir / "deployment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (out_dir / "latency_per_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "astar_ms", "policy_ms", "success", "expert_len", "pred_len"])
        writer.writeheader()
        writer.writerows(per_case_rows)

    table_rows = [
        {"metric": "astar_mean_ms", "value": astar_stats["mean"]},
        {"metric": "astar_p50_ms", "value": astar_stats["p50"]},
        {"metric": "astar_p90_ms", "value": astar_stats["p90"]},
        {"metric": "astar_p99_ms", "value": astar_stats["p99"]},
        {"metric": "policy_mean_ms", "value": policy_stats["mean"]},
        {"metric": "policy_p50_ms", "value": policy_stats["p50"]},
        {"metric": "policy_p90_ms", "value": policy_stats["p90"]},
        {"metric": "policy_p99_ms", "value": policy_stats["p99"]},
        {"metric": "astar_qps", "value": astar_qps},
        {"metric": "policy_qps", "value": policy_qps},
        {"metric": "model_file_size_mb", "value": file_size_mb},
        {"metric": "model_parameter_count", "value": float(param_count)},
        {"metric": "model_parameter_memory_mb_fp32", "value": param_mem_mb},
    ]

    with (out_dir / "deployment_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(table_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
