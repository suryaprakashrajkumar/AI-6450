from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .gridworld import ACTION_DELTAS, astar_path, extract_local_patch, move_cost, relative_goal
from .train_torch_maze import MultiTaskMLP


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run saved Torch policy on multiple maze test cases")
    p.add_argument("--model", type=str, default="models/torch/torch_mlp_small.pt")
    p.add_argument("--episode-data", type=str, default="data/maze_small/episode_data.npz")
    p.add_argument("--trajectories", type=str, default="data/maze_small/expert_trajectories.jsonl")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--num-cases", type=int, default=4)
    p.add_argument("--max-revisit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="artifacts/inference")
    return p


def split_code(name: str) -> Optional[int]:
    return {"train": 0, "val": 1, "test": 2}.get(name)


def path_cost(path: List[Tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(move_cost(path[i], path[i + 1]) for i in range(len(path) - 1)))


def rollout(
    model: MultiTaskMLP,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    patch_size: int,
    device: torch.device,
    max_revisit: int,
) -> Dict[str, object]:
    model.eval()
    pos = start
    path: List[Tuple[int, int]] = [start]
    visits: Dict[Tuple[int, int], int] = {start: 1}
    decision_ms: List[float] = []

    max_steps = max(50, 4 * (grid.shape[0] + grid.shape[1]))
    for _ in range(max_steps):
        if pos == goal:
            return {
                "success": True,
                "collision": False,
                "timeout": False,
                "path": path,
                "path_cost": path_cost(path),
                "mean_decision_time_ms": float(np.mean(decision_ms)) if decision_ms else 0.0,
            }

        patch = extract_local_patch(grid, pos, patch_size).astype(np.float32).reshape(1, -1)
        rg = relative_goal(pos, goal, grid.shape).astype(np.float32).reshape(1, -1)
        agent = np.array([[pos[0] / max(grid.shape[0] - 1, 1), pos[1] / max(grid.shape[1] - 1, 1)]], dtype=np.float32)
        feat = np.concatenate([patch, rg, agent], axis=1)

        xb = torch.from_numpy(feat).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze(0)
        decision_ms.append((time.perf_counter() - t0) * 1000.0)

        ranked = list(np.argsort(probs)[::-1])

        def valid(a: int) -> Tuple[bool, Tuple[int, int]]:
            dr, dc = ACTION_DELTAS[a]
            nxt = (pos[0] + dr, pos[1] + dc)
            if not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]):
                return False, nxt
            if grid[nxt[0], nxt[1]] == 1:
                return False, nxt
            return True, nxt

        chosen = None
        chosen_next = None
        for a in ranked:
            ok, nxt = valid(int(a))
            if ok and visits.get(nxt, 0) <= max_revisit:
                chosen = int(a)
                chosen_next = nxt
                break
        if chosen is None:
            for a in ranked:
                ok, nxt = valid(int(a))
                if ok:
                    chosen = int(a)
                    chosen_next = nxt
                    break

        if chosen_next is None:
            return {
                "success": False,
                "collision": True,
                "timeout": False,
                "path": path,
                "path_cost": path_cost(path),
                "mean_decision_time_ms": float(np.mean(decision_ms)) if decision_ms else 0.0,
            }

        pos = chosen_next
        path.append(pos)
        visits[pos] = visits.get(pos, 0) + 1

    return {
        "success": False,
        "collision": False,
        "timeout": True,
        "path": path,
        "path_cost": path_cost(path),
        "mean_decision_time_ms": float(np.mean(decision_ms)) if decision_ms else 0.0,
    }


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device)
    model = MultiTaskMLP(in_dim=int(ckpt["in_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    patch_size = int(ckpt.get("patch_size", 15))

    ep_data = np.load(args.episode_data)
    grids = ep_data["episode_grid"]
    starts = ep_data["episode_start"]
    goals = ep_data["episode_goal"]
    splits = ep_data["episode_split"]

    trajectories: List[Dict[str, object]] = []
    with open(args.trajectories, "r", encoding="utf-8") as f:
        for line in f:
            trajectories.append(json.loads(line))

    candidates = np.arange(grids.shape[0])
    s_code = split_code(args.split)
    if s_code is not None:
        candidates = candidates[splits == s_code]
    if candidates.size == 0:
        raise ValueError("No episodes found for selected split")

    n = min(args.num_cases, int(candidates.size))
    chosen = rng.choice(candidates, size=n, replace=False)

    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 11))
    axes = np.array(axes).reshape(rows, cols)

    results: List[Dict[str, object]] = []

    for k in range(rows * cols):
        ax = axes.flat[k]
        if k >= n:
            ax.axis("off")
            continue

        ep = int(chosen[k])
        grid = grids[ep]
        start = tuple(map(int, starts[ep]))
        goal = tuple(map(int, goals[ep]))

        run = rollout(model, grid, start, goal, patch_size, device, args.max_revisit)
        pred_path = run["path"]

        expert = [tuple(p) for p in trajectories[ep]["path"]]
        opt_cost = path_cost(expert)
        ratio = float(run["path_cost"] / max(opt_cost, 1e-8)) if bool(run["success"]) else float("nan")

        astar_t0 = time.perf_counter()
        astar_p = astar_path(grid, start, goal)
        astar_ms = (time.perf_counter() - astar_t0) * 1000.0

        ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        p = np.asarray(pred_path)
        ax.plot(p[:, 1], p[:, 0], color="tab:blue", linewidth=2, label="model")
        if astar_p is not None and len(astar_p) >= 2:
            a = np.asarray(astar_p)
            ax.plot(a[:, 1], a[:, 0], color="tab:orange", linewidth=1.5, alpha=0.8, label="A*")

        ax.scatter([start[1]], [start[0]], c="tab:green", s=60, marker="o")
        ax.scatter([goal[1]], [goal[0]], c="tab:red", s=70, marker="*")
        status = "OK" if bool(run["success"]) else ("TO" if bool(run["timeout"]) else "COL")
        ax.set_title(
            f"ep={ep} {status} steps={len(pred_path)-1}\n"
            f"ratio={ratio:.2f} model={float(run['mean_decision_time_ms']):.3f}ms A*={astar_ms:.3f}ms",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        results.append(
            {
                "episode_id": ep,
                "split": int(splits[ep]),
                "success": bool(run["success"]),
                "collision": bool(run["collision"]),
                "timeout": bool(run["timeout"]),
                "pred_path_steps": int(len(pred_path) - 1),
                "pred_path_cost": float(run["path_cost"]),
                "optimal_path_cost": float(opt_cost),
                "optimality_ratio": ratio,
                "mean_model_decision_time_ms": float(run["mean_decision_time_ms"]),
                "astar_solve_time_ms": float(astar_ms),
            }
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    img_path = out_dir / "policy_run_4cases.png"
    json_path = out_dir / "policy_run_4cases.json"
    fig.savefig(img_path, dpi=180)
    plt.close(fig)

    summary = {
        "num_cases": n,
        "success_rate": float(np.mean([float(r["success"]) for r in results])) if results else 0.0,
        "cases": results,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Batch policy run complete")
    print(json.dumps(summary, indent=2))
    print(f"Saved image: {img_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
