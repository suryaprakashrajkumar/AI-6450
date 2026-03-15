from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.eval import evaluate_model


def _draw_line_cells(start: np.ndarray, goal: np.ndarray) -> list[tuple[int, int]]:
    r0, c0 = int(start[0]), int(start[1])
    r1, c1 = int(goal[0]), int(goal[1])
    cells: list[tuple[int, int]] = []
    steps = max(abs(r1 - r0), abs(c1 - c0), 1)
    for t in range(steps + 1):
        a = t / steps
        r = int(round(r0 + a * (r1 - r0)))
        c = int(round(c0 + a * (c1 - c0)))
        cells.append((r, c))
    return cells


def _apply_corridor_attack(grid: np.ndarray, start: np.ndarray, goal: np.ndarray, strength: int, rng: np.random.Generator) -> np.ndarray:
    out = grid.copy()
    cells = _draw_line_cells(start, goal)
    if len(cells) <= 2:
        return out

    frac = {1: 0.25, 2: 0.50, 3: 0.75}.get(strength, 0.0)
    k = max(1, int((len(cells) - 2) * frac))
    candidate = cells[1:-1]
    rng.shuffle(candidate)
    for r, c in candidate[:k]:
        out[r, c] = 1

    out[int(start[0]), int(start[1])] = 0
    out[int(goal[0]), int(goal[1])] = 0
    return out


def _build_adv_dataset(base_rollout_path: str, out_path: str, strength: int, seed: int) -> None:
    payload = np.load(base_rollout_path)
    grids = payload["grids"]
    starts = payload["starts"]
    goals_full = payload["goals_full"]

    rng = np.random.default_rng(seed + 100 * strength)
    attacked = np.zeros_like(grids)
    for i in range(len(grids)):
        attacked[i] = _apply_corridor_attack(grids[i], starts[i], goals_full[i], strength, rng)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, grids=attacked, starts=starts, goals_full=goals_full)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_improved.json")
    parser.add_argument("--model", type=str, default="models/final/imitation_policy_improved.pth")
    parser.add_argument("--test_rollout", type=str, default="data/processed/test_rollout.npz")
    parser.add_argument("--test_steps", type=str, default="data/processed/test_samples.npz")
    parser.add_argument("--out_dir", type=str, default="logs/adversarial")
    parser.add_argument("--rollout_limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []

    clean_metrics = evaluate_model(
        rollout_npz_path=args.test_rollout,
        step_npz_path=args.test_steps,
        model_path=args.model,
        config_path=args.config,
        metrics_output=str(out_dir / "clean_metrics.json"),
        compute_step_accuracy=False,
        rollout_limit=args.rollout_limit,
    )
    rows.append({"scenario": "clean", "strength": 0, **clean_metrics})

    for strength in [1, 2, 3]:
        ds_path = out_dir / f"attack_s{strength}.npz"
        _build_adv_dataset(args.test_rollout, str(ds_path), strength, args.seed)
        metrics = evaluate_model(
            rollout_npz_path=str(ds_path),
            step_npz_path=args.test_steps,
            model_path=args.model,
            config_path=args.config,
            metrics_output=str(out_dir / f"attack_s{strength}_metrics.json"),
            compute_step_accuracy=False,
            rollout_limit=args.rollout_limit,
        )
        row = {"scenario": "corridor_block", "strength": strength, **metrics}
        rows.append(row)
        print(f"attack strength={strength} path_validity={float(metrics['path_validity']):.4f} exact_match={float(metrics['exact_match']):.4f}")

    csv_path = out_dir / "adversarial_results.csv"
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
        "threat_model": {
            "white_box": "knows map geometry and blocks straight-line corridor",
            "gray_box": "knows start/goal, partial knowledge of policy behavior",
            "black_box": "can inject obstacle perturbations without model access"
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
