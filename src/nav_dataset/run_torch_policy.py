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
    p = argparse.ArgumentParser(description="Run saved Torch policy on maze and save inference artifacts")
    p.add_argument("--model", type=str, default="models/torch/torch_mlp_small.pt")
    p.add_argument("--episode-data", type=str, default="data/maze_small/episode_data.npz")
    p.add_argument("--trajectories", type=str, default="data/maze_small/expert_trajectories.jsonl")
    p.add_argument("--episode-id", type=int, default=-1, help="Episode id to run. -1 selects automatically.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--find-success", action="store_true", help="Search episodes until a successful rollout is found")
    p.add_argument("--max-tries", type=int, default=100)
    p.add_argument("--max-revisit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="artifacts/inference")
    return p


def split_code(name: str) -> Optional[int]:
    if name == "train":
        return 0
    if name == "val":
        return 1
    if name == "test":
        return 2
    return None


def path_cost(path: List[Tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(move_cost(path[i], path[i + 1]) for i in range(len(path) - 1)))


def rollout_with_trace(
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
    visit_counts: Dict[Tuple[int, int], int] = {start: 1}
    trace: List[Dict[str, object]] = []
    decision_times: List[float] = []

    max_steps = max(50, 4 * (grid.shape[0] + grid.shape[1]))
    for step in range(max_steps):
        if pos == goal:
            return {
                "success": True,
                "collision": False,
                "timeout": False,
                "path": path,
                "trace": trace,
                "path_cost": path_cost(path),
                "mean_decision_time_ms": float(np.mean(decision_times) * 1000.0) if decision_times else 0.0,
            }

        patch = extract_local_patch(grid, pos, patch_size).astype(np.float32).reshape(1, -1)
        rg = relative_goal(pos, goal, grid.shape).astype(np.float32).reshape(1, -1)
        agent = np.array([[pos[0] / max(grid.shape[0] - 1, 1), pos[1] / max(grid.shape[1] - 1, 1)]], dtype=np.float32)
        feat = np.concatenate([patch, rg, agent], axis=1)

        xb = torch.from_numpy(feat).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, cost_pred_n = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze(0)
            pred_cost_n = float(cost_pred_n.cpu().numpy().squeeze())
        dt = (time.perf_counter() - t0) * 1000.0
        decision_times.append(dt / 1000.0)

        ranked = list(np.argsort(probs)[::-1])

        def valid(a: int) -> Tuple[bool, Tuple[int, int]]:
            dr, dc = ACTION_DELTAS[a]
            nxt = (pos[0] + dr, pos[1] + dc)
            if not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]):
                return False, nxt
            if grid[nxt[0], nxt[1]] == 1:
                return False, nxt
            return True, nxt

        chosen: Optional[int] = None
        chosen_next: Optional[Tuple[int, int]] = None

        for a in ranked:
            ok, nxt = valid(int(a))
            if ok and visit_counts.get(nxt, 0) <= max_revisit:
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

        if chosen is None or chosen_next is None:
            return {
                "success": False,
                "collision": True,
                "timeout": False,
                "path": path,
                "trace": trace,
                "path_cost": path_cost(path),
                "mean_decision_time_ms": float(np.mean(np.array(decision_times) * 1000.0)) if decision_times else 0.0,
            }

        trace.append(
            {
                "step": step,
                "position": [int(pos[0]), int(pos[1])],
                "chosen_action": int(chosen),
                "chosen_prob": float(probs[chosen]),
                "predicted_cost_norm": pred_cost_n,
                "decision_time_ms": float(dt),
            }
        )

        pos = chosen_next
        path.append(pos)
        visit_counts[pos] = visit_counts.get(pos, 0) + 1

    return {
        "success": False,
        "collision": False,
        "timeout": True,
        "path": path,
        "trace": trace,
        "path_cost": path_cost(path),
        "mean_decision_time_ms": float(np.mean(np.array(decision_times) * 1000.0)) if decision_times else 0.0,
    }


def save_plot(
    out_file: Path,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    pred_path: List[Tuple[int, int]],
    astar_path_pts: Optional[List[Tuple[int, int]]],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)

    p = np.asarray(pred_path)
    ax.plot(p[:, 1], p[:, 0], color="tab:blue", linewidth=2, label="model")

    if astar_path_pts is not None and len(astar_path_pts) >= 2:
        a = np.asarray(astar_path_pts)
        ax.plot(a[:, 1], a[:, 0], color="tab:orange", linewidth=1.5, alpha=0.8, label="A*")

    ax.scatter([start[1]], [start[0]], c="tab:green", s=70, marker="o", label="start")
    ax.scatter([goal[1]], [goal[0]], c="tab:red", s=90, marker="*", label="goal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


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

    chosen_episode = int(args.episode_id) if args.episode_id >= 0 else int(rng.choice(candidates))

    best_result: Optional[Dict[str, object]] = None
    best_ep = chosen_episode

    tries = min(args.max_tries, int(candidates.size)) if args.find_success else 1
    trial_eps = [chosen_episode]
    if args.find_success:
        shuffled = rng.permutation(candidates)
        trial_eps = [int(e) for e in shuffled[:tries]]

    for ep in trial_eps:
        grid = grids[ep]
        start = tuple(map(int, starts[ep]))
        goal = tuple(map(int, goals[ep]))
        res = rollout_with_trace(model, grid, start, goal, patch_size, device, max_revisit=args.max_revisit)
        best_result = res
        best_ep = ep
        if bool(res["success"]):
            break

    assert best_result is not None
    grid = grids[best_ep]
    start = tuple(map(int, starts[best_ep]))
    goal = tuple(map(int, goals[best_ep]))
    expert = [tuple(p) for p in trajectories[best_ep]["path"]]
    opt_cost = path_cost(expert)
    pred_cost = float(best_result["path_cost"])

    astar_t0 = time.perf_counter()
    astar_p = astar_path(grid, start, goal)
    astar_dt = (time.perf_counter() - astar_t0) * 1000.0

    out_json = out_dir / f"policy_run_episode_{best_ep}.json"
    out_png = out_dir / f"policy_run_episode_{best_ep}.png"

    report = {
        "episode_id": int(best_ep),
        "split": int(splits[best_ep]),
        "success": bool(best_result["success"]),
        "collision": bool(best_result["collision"]),
        "timeout": bool(best_result["timeout"]),
        "pred_path_steps": int(len(best_result["path"]) - 1),
        "pred_path_cost": pred_cost,
        "optimal_path_cost": opt_cost,
        "optimality_ratio": float(pred_cost / max(opt_cost, 1e-8)) if bool(best_result["success"]) else float("nan"),
        "mean_model_decision_time_ms": float(best_result["mean_decision_time_ms"]),
        "astar_solve_time_ms": float(astar_dt),
        "trace": best_result["trace"],
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_plot(out_png, grid, start, goal, best_result["path"], astar_p)

    print("Policy run complete")
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))
    print(f"Saved report: {out_json}")
    print(f"Saved plot: {out_png}")

    print("\nFirst rollout steps:")
    for row in report["trace"][:10]:
        print(
            f"step={row['step']} pos={row['position']} action={row['chosen_action']} "
            f"p={row['chosen_prob']:.3f} dt_ms={row['decision_time_ms']:.3f}"
        )


if __name__ == "__main__":
    main()
