from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np

from src.eval import evaluate_model


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ensure_start_goal_free(grid: np.ndarray, start: np.ndarray, goal: np.ndarray) -> None:
    grid[int(start[0]), int(start[1])] = 0
    grid[int(goal[0]), int(goal[1])] = 0


def _scenario_flip_noise(grid: np.ndarray, start: np.ndarray, goal: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    p = {1: 0.03, 2: 0.06, 3: 0.10}.get(severity, 0.0)
    if p <= 0.0:
        return grid
    mask = rng.random((10, 10)) < p
    out = grid.copy()
    out[mask] = 1 - out[mask]
    _ensure_start_goal_free(out, start, goal)
    return out


def _scenario_mask_dropout(grid: np.ndarray, start: np.ndarray, goal: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    p = {1: 0.03, 2: 0.07, 3: 0.12}.get(severity, 0.0)
    if p <= 0.0:
        return grid
    out = grid.copy()
    # Dropout here means unknown cells pessimistically treated as blocked.
    mask = rng.random((10, 10)) < p
    out[mask] = 1
    _ensure_start_goal_free(out, start, goal)
    return out


def _scenario_occlusion(grid: np.ndarray, start: np.ndarray, goal: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    k = {1: 2, 2: 3, 3: 4}.get(severity, 0)
    if k <= 0:
        return grid
    out = grid.copy()
    r = int(rng.integers(0, 10 - k + 1))
    c = int(rng.integers(0, 10 - k + 1))
    out[r : r + k, c : c + k] = 1
    _ensure_start_goal_free(out, start, goal)
    return out


def _scenario_density_shift(grid: np.ndarray, start: np.ndarray, goal: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    target = {1: 0.35, 2: 0.45, 3: 0.55}.get(severity, 0.0)
    if target <= 0:
        return grid
    out = grid.copy()
    cur_density = float(out.mean())
    if cur_density >= target:
        _ensure_start_goal_free(out, start, goal)
        return out
    add_ratio = min(1.0, (target - cur_density) / max(1e-6, (1.0 - cur_density)))
    free_mask = out == 0
    add_mask = (rng.random((10, 10)) < add_ratio) & free_mask
    out[add_mask] = 1
    _ensure_start_goal_free(out, start, goal)
    return out


def _build_dataset(
    base_rollout_path: str,
    out_path: str,
    transform: Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.random.Generator], np.ndarray],
    severity: int,
    seed: int,
) -> None:
    payload = np.load(base_rollout_path)
    grids = payload["grids"]
    starts = payload["starts"]
    goals_full = payload["goals_full"]

    rng = _rng(seed + severity * 17)
    perturbed = np.zeros_like(grids)

    for i in range(len(grids)):
        perturbed[i] = transform(grids[i].copy(), starts[i], goals_full[i], severity, rng)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, grids=perturbed, starts=starts, goals_full=goals_full)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_improved.json")
    parser.add_argument("--model", type=str, default="models/final/imitation_policy_improved.pth")
    parser.add_argument("--test_rollout", type=str, default="data/processed/test_rollout.npz")
    parser.add_argument("--test_steps", type=str, default="data/processed/test_samples.npz")
    parser.add_argument("--out_dir", type=str, default="logs/stress_tests")
    parser.add_argument("--rollout_limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.random.Generator], np.ndarray]] = {
        "flip_noise": _scenario_flip_noise,
        "mask_dropout": _scenario_mask_dropout,
        "occlusion": _scenario_occlusion,
        "density_shift": _scenario_density_shift,
    }

    rows: list[dict[str, float | str]] = []

    # Clean baseline
    clean_metrics = evaluate_model(
        rollout_npz_path=args.test_rollout,
        step_npz_path=args.test_steps,
        model_path=args.model,
        config_path=args.config,
        metrics_output=str(out_dir / "clean_metrics.json"),
        compute_step_accuracy=False,
        rollout_limit=args.rollout_limit,
    )
    rows.append({"scenario": "clean", "severity": 0, **clean_metrics})

    for scenario, transform in scenarios.items():
        for severity in [1, 2, 3]:
            ds_path = out_dir / f"{scenario}_s{severity}.npz"
            _build_dataset(
                base_rollout_path=args.test_rollout,
                out_path=str(ds_path),
                transform=transform,
                severity=severity,
                seed=args.seed,
            )
            metrics = evaluate_model(
                rollout_npz_path=str(ds_path),
                step_npz_path=args.test_steps,
                model_path=args.model,
                config_path=args.config,
                metrics_output=str(out_dir / f"{scenario}_s{severity}_metrics.json"),
                compute_step_accuracy=False,
                rollout_limit=args.rollout_limit,
            )
            row = {"scenario": scenario, "severity": severity, **metrics}
            rows.append(row)
            print(f"{scenario} severity={severity} path_validity={float(metrics['path_validity']):.4f} policy_ms={float(metrics['policy_ms_per_query']):.4f}")

    csv_path = out_dir / "stress_results.csv"
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_rows": len(rows),
        "csv": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
