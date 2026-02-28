from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .gridworld import ACTION_DELTAS, astar_path, extract_local_patch, move_cost, relative_goal


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PyTorch MLP on maze data and compare to A*")
    p.add_argument("--dataset", type=str, default="data_maze_small/navigation_dataset.npz")
    p.add_argument("--episode-data", type=str, default="data_maze_small/episode_data.npz")
    p.add_argument("--trajectories", type=str, default="data_maze_small/expert_trajectories.jsonl")
    p.add_argument("--output", type=str, default="torch_runs")
    p.add_argument("--artifacts", type=str, default="artifacts")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=40000)
    p.add_argument("--max-eval-samples", type=int, default=12000)
    p.add_argument("--show-inference", type=int, default=12)
    p.add_argument("--max-test-episodes", type=int, default=120)
    p.add_argument(
        "--action-loss",
        type=str,
        default="hybrid",
        choices=["mse", "ce", "hybrid"],
        help="Action training loss type. hybrid = CE + 0.25*MSE(one-hot).",
    )
    p.add_argument("--action-loss-weight", type=float, default=1.0)
    p.add_argument("--cost-loss-weight", type=float, default=1.0)
    p.add_argument("--use-class-weights", action="store_true")
    p.add_argument("--max-revisit", type=int, default=2, help="Max revisit count per cell during rollout")
    return p


class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, 8)
        self.cost_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        action_logits = self.action_head(h)
        cost = self.cost_head(h).squeeze(-1)
        return action_logits, cost


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_indices(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if idx.size <= cap:
        return idx
    return rng.choice(idx, size=cap, replace=False)


def build_features(data: np.lib.npyio.NpzFile) -> np.ndarray:
    patch = data["local_patch"].astype(np.float32).reshape(data["local_patch"].shape[0], -1)
    rel_goal_vec = data["relative_goal"].astype(np.float32)
    pos = data["agent_position"].astype(np.float32)

    pos[:, 0] /= max(float(np.max(pos[:, 0])), 1.0)
    pos[:, 1] /= max(float(np.max(pos[:, 1])), 1.0)

    return np.concatenate([patch, rel_goal_vec, pos], axis=1).astype(np.float32)


def to_loader(
    x: np.ndarray,
    y_action: np.ndarray,
    y_cost_norm: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y_action.astype(np.int64)),
        torch.from_numpy(y_cost_norm.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_step_metrics(
    model: nn.Module,
    x: np.ndarray,
    y_action: np.ndarray,
    y_cost: np.ndarray,
    cost_scale: float,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(x).to(device)
        logits, cost_norm = model(xb)
        pred_action = torch.argmax(logits, dim=1).cpu().numpy()
        pred_cost = (cost_norm.cpu().numpy() * cost_scale).astype(np.float32)

    acc = float(np.mean(pred_action == y_action))
    mae = float(np.mean(np.abs(pred_cost - y_cost)))
    rmse = float(np.sqrt(np.mean((pred_cost - y_cost) ** 2)))
    mse = float(np.mean((pred_cost - y_cost) ** 2))
    var = float(np.var(y_cost))
    r2 = float(1.0 - mse / max(var, 1e-8))
    return {
        "action_accuracy": acc,
        "cost_mae": mae,
        "cost_rmse": rmse,
        "cost_r2": r2,
    }


def path_cost(path: List[Tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    out = 0.0
    for i in range(len(path) - 1):
        out += move_cost(path[i], path[i + 1])
    return float(out)


def rollout_policy(
    model: nn.Module,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    patch_size: int,
    device: torch.device,
    max_revisit: int,
) -> Dict[str, float | bool]:
    model.eval()
    pos = start
    visited_path: List[Tuple[int, int]] = [start]
    visit_counts: Dict[Tuple[int, int], int] = {start: 1}
    decision_times: List[float] = []

    max_steps = max(50, 4 * (grid.shape[0] + grid.shape[1]))
    for _ in range(max_steps):
        if pos == goal:
            return {
                "success": True,
                "collision": False,
                "timeout": False,
                "path_cost": path_cost(visited_path),
                "steps": len(visited_path) - 1,
                "mean_decision_time_s": float(np.mean(decision_times)) if decision_times else 0.0,
            }

        patch = extract_local_patch(grid, pos, patch_size).astype(np.float32).reshape(1, -1)
        rg = relative_goal(pos, goal, grid.shape).astype(np.float32).reshape(1, -1)
        agent = np.array([[pos[0] / max(grid.shape[0] - 1, 1), pos[1] / max(grid.shape[1] - 1, 1)]], dtype=np.float32)
        feat = np.concatenate([patch, rg, agent], axis=1)

        x = torch.from_numpy(feat).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(x)
            logits_np = logits.squeeze(0).cpu().numpy()

            def is_valid_action(a: int) -> Tuple[bool, Tuple[int, int]]:
                dr_, dc_ = ACTION_DELTAS[a]
                nxt_ = (pos[0] + dr_, pos[1] + dc_)
                if not (0 <= nxt_[0] < grid.shape[0] and 0 <= nxt_[1] < grid.shape[1]):
                    return False, nxt_
                if grid[nxt_[0], nxt_[1]] == 1:
                    return False, nxt_
                return True, nxt_

            ranked = list(np.argsort(logits_np)[::-1])
            action: Optional[int] = None

            # Pass 1: valid and not over-visited
            for a in ranked:
                ok, nxt_tmp = is_valid_action(int(a))
                if ok and visit_counts.get(nxt_tmp, 0) <= max_revisit:
                    action = int(a)
                    break

            # Pass 2: any valid action
            if action is None:
                for a in ranked:
                    ok, _ = is_valid_action(int(a))
                    if ok:
                        action = int(a)
                        break

            if action is None:
                return {
                    "success": False,
                    "collision": True,
                    "timeout": False,
                    "path_cost": path_cost(visited_path),
                    "steps": len(visited_path) - 1,
                    "mean_decision_time_s": float(np.mean(decision_times)) if decision_times else 0.0,
                }
        dt = time.perf_counter() - t0
        decision_times.append(dt)

        dr, dc = ACTION_DELTAS[action]
        nxt = (pos[0] + dr, pos[1] + dc)

        if not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]):
            return {
                "success": False,
                "collision": True,
                "timeout": False,
                "path_cost": path_cost(visited_path),
                "steps": len(visited_path) - 1,
                "mean_decision_time_s": float(np.mean(decision_times)),
            }
        if grid[nxt[0], nxt[1]] == 1:
            return {
                "success": False,
                "collision": True,
                "timeout": False,
                "path_cost": path_cost(visited_path),
                "steps": len(visited_path) - 1,
                "mean_decision_time_s": float(np.mean(decision_times)),
            }

        visited_path.append(nxt)
        visit_counts[nxt] = visit_counts.get(nxt, 0) + 1
        pos = nxt

    return {
        "success": False,
        "collision": False,
        "timeout": True,
        "path_cost": path_cost(visited_path),
        "steps": len(visited_path) - 1,
        "mean_decision_time_s": float(np.mean(decision_times)) if decision_times else 0.0,
    }


def plot_training_curves(history: Dict[str, List[float]], path: Path) -> None:
    epochs = np.arange(1, len(history["train_total_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, history["train_total_loss"], label="train total")
    axes[0, 0].plot(epochs, history["val_total_loss"], label="val total")
    axes[0, 0].set_title("Total MSE loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_action_mse"], label="train action MSE")
    axes[0, 1].plot(epochs, history["val_action_mse"], label="val action MSE")
    axes[0, 1].set_title("Action-head MSE")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["train_cost_mse"], label="train cost MSE")
    axes[1, 0].plot(epochs, history["val_cost_mse"], label="val cost MSE")
    axes[1, 0].set_title("Cost-head MSE")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history["val_action_acc"], label="val action acc")
    axes[1, 1].plot(epochs, history["val_cost_rmse"], label="val cost RMSE")
    axes[1, 1].set_title("Validation metrics")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_kpi_comparison(kpis: Dict[str, Dict[str, float]], path: Path) -> None:
    methods = ["A*", "Torch MLP"]
    success = [kpis["astar"]["success_rate"], kpis["mlp"]["success_rate"]]
    opt_ratio = [kpis["astar"]["optimality_ratio"], kpis["mlp"]["optimality_ratio"]]
    dt_ms = [kpis["astar"]["decision_time_ms"], kpis["mlp"]["decision_time_ms"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(methods, success)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Success rate")

    axes[1].bar(methods, opt_ratio)
    axes[1].set_title("Optimality ratio (lower is better)")

    axes[2].bar(methods, dt_ms)
    axes[2].set_title("Decision time (ms)")

    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_additional_metrics(additional: Dict[str, float], opt_ratios: List[float], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    names = ["collision_rate", "timeout_rate", "test_action_acc", "test_cost_rmse"]
    vals = [additional[n] for n in names]
    axes[0].bar(names, vals)
    axes[0].set_title("Additional metrics")
    axes[0].tick_params(axis="x", rotation=25)

    if len(opt_ratios) > 0:
        axes[1].hist(opt_ratios, bins=20)
    axes[1].set_title("MLP optimality ratio distribution")
    axes[1].set_xlabel("ratio")

    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output)
    art_dir = Path(args.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    x = build_features(data)
    y_action = data["action_id"].astype(np.int64)
    y_cost = data["cost_to_goal"].astype(np.float32)
    split = data["split"].astype(np.int64)

    train_idx = np.where(split == 0)[0]
    val_idx = np.where(split == 1)[0]
    test_idx = np.where(split == 2)[0]
    train_idx = sample_indices(train_idx, args.max_train_samples, rng)
    val_idx = sample_indices(val_idx, args.max_eval_samples, rng)
    test_idx = sample_indices(test_idx, args.max_eval_samples, rng)

    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    y_act_train, y_act_val, y_act_test = y_action[train_idx], y_action[val_idx], y_action[test_idx]
    y_cost_train, y_cost_val, y_cost_test = y_cost[train_idx], y_cost[val_idx], y_cost[test_idx]

    cost_scale = float(np.max(y_cost_train)) if float(np.max(y_cost_train)) > 0 else 1.0
    y_cost_train_n = y_cost_train / cost_scale
    y_cost_val_n = y_cost_val / cost_scale

    train_loader = to_loader(x_train, y_act_train, y_cost_train_n, args.batch_size, shuffle=True)
    val_loader = to_loader(x_val, y_act_val, y_cost_val_n, args.batch_size, shuffle=False)

    model = MultiTaskMLP(in_dim=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()

    class_weights = None
    if args.use_class_weights:
        counts = np.bincount(y_act_train, minlength=8).astype(np.float32)
        inv = 1.0 / np.clip(counts, 1.0, None)
        inv = inv / np.mean(inv)
        class_weights = torch.from_numpy(inv).to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    history: Dict[str, List[float]] = {
        "train_total_loss": [],
        "val_total_loss": [],
        "train_action_mse": [],
        "val_action_mse": [],
        "train_cost_mse": [],
        "val_cost_mse": [],
        "val_action_acc": [],
        "val_cost_rmse": [],
    }

    best_val = float("inf")
    best_path = out_dir / "torch_mlp_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_total = 0.0
        tr_act = 0.0
        tr_cost = 0.0
        tr_n = 0

        for xb, ya, yc in train_loader:
            xb = xb.to(device)
            ya = ya.to(device)
            yc = yc.to(device)

            logits, cost_pred = model(xb)
            ya_onehot = torch.zeros((ya.shape[0], 8), device=device)
            ya_onehot.scatter_(1, ya.unsqueeze(1), 1.0)

            if args.action_loss == "mse":
                loss_act = mse(torch.softmax(logits, dim=1), ya_onehot)
            elif args.action_loss == "ce":
                loss_act = ce(logits, ya)
            else:
                loss_act = ce(logits, ya) + 0.25 * mse(torch.softmax(logits, dim=1), ya_onehot)
            loss_cost = mse(cost_pred, yc)
            loss = args.action_loss_weight * loss_act + args.cost_loss_weight * loss_cost

            opt.zero_grad()
            loss.backward()
            opt.step()

            b = xb.shape[0]
            tr_total += float(loss.item()) * b
            tr_act += float(loss_act.item()) * b
            tr_cost += float(loss_cost.item()) * b
            tr_n += b

        model.eval()
        va_total = 0.0
        va_act = 0.0
        va_cost = 0.0
        va_n = 0
        val_acc_sum = 0.0
        val_rmse_sum = 0.0

        with torch.no_grad():
            for xb, ya, yc in val_loader:
                xb = xb.to(device)
                ya = ya.to(device)
                yc = yc.to(device)
                logits, cost_pred = model(xb)

                ya_onehot = torch.zeros((ya.shape[0], 8), device=device)
                ya_onehot.scatter_(1, ya.unsqueeze(1), 1.0)

                if args.action_loss == "mse":
                    loss_act = mse(torch.softmax(logits, dim=1), ya_onehot)
                elif args.action_loss == "ce":
                    loss_act = ce(logits, ya)
                else:
                    loss_act = ce(logits, ya) + 0.25 * mse(torch.softmax(logits, dim=1), ya_onehot)
                loss_cost = mse(cost_pred, yc)
                loss = args.action_loss_weight * loss_act + args.cost_loss_weight * loss_cost

                b = xb.shape[0]
                va_total += float(loss.item()) * b
                va_act += float(loss_act.item()) * b
                va_cost += float(loss_cost.item()) * b
                va_n += b

                pred_a = torch.argmax(logits, dim=1)
                val_acc_sum += float((pred_a == ya).float().sum().item())

                pred_cost = cost_pred * cost_scale
                true_cost = yc * cost_scale
                val_rmse_sum += float(torch.sum((pred_cost - true_cost) ** 2).item())

        tr_total /= max(tr_n, 1)
        tr_act /= max(tr_n, 1)
        tr_cost /= max(tr_n, 1)

        va_total /= max(va_n, 1)
        va_act /= max(va_n, 1)
        va_cost /= max(va_n, 1)
        va_acc = val_acc_sum / max(va_n, 1)
        va_rmse = float(np.sqrt(val_rmse_sum / max(va_n, 1)))

        history["train_total_loss"].append(tr_total)
        history["val_total_loss"].append(va_total)
        history["train_action_mse"].append(tr_act)
        history["val_action_mse"].append(va_act)
        history["train_cost_mse"].append(tr_cost)
        history["val_cost_mse"].append(va_cost)
        history["val_action_acc"].append(va_acc)
        history["val_cost_rmse"].append(va_rmse)

        if va_total < best_val:
            best_val = va_total
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_dim": int(x_train.shape[1]),
                    "cost_scale": cost_scale,
                    "patch_size": int(data["local_patch"].shape[1]),
                    "action_loss": args.action_loss,
                    "max_revisit": args.max_revisit,
                },
                best_path,
            )

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={tr_total:.4f} val_loss={va_total:.4f} "
                f"val_acc={va_acc:.3f} val_cost_rmse={va_rmse:.3f}"
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_metrics = evaluate_step_metrics(model, x_val, y_act_val, y_cost_val, cost_scale=cost_scale, device=device)
    test_metrics = evaluate_step_metrics(model, x_test, y_act_test, y_cost_test, cost_scale=cost_scale, device=device)

    # Print sample inferences.
    show_n = min(args.show_inference, x_test.shape[0])
    show_idx = rng.choice(np.arange(x_test.shape[0]), size=show_n, replace=False)
    with torch.no_grad():
        logits, cost_n = model(torch.from_numpy(x_test[show_idx]).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_a = np.argmax(probs, axis=1)
        pred_c = (cost_n.cpu().numpy() * cost_scale).astype(np.float32)

    print("\nSample inferences (Torch MLP, test split):")
    for i, li in enumerate(show_idx):
        gi = int(test_idx[li])
        print(
            f"idx={gi} | action true/pred={int(y_act_test[li])}/{int(pred_a[i])} "
            f"(conf={float(np.max(probs[i])):.3f}) | "
            f"cost true/pred={float(y_cost_test[li]):.2f}/{float(pred_c[i]):.2f}"
        )

    # Episode-level KPI comparison against A*.
    ep_data = np.load(args.episode_data)
    ep_grids = ep_data["episode_grid"]
    ep_starts = ep_data["episode_start"]
    ep_goals = ep_data["episode_goal"]
    ep_split = ep_data["episode_split"]

    trajectories: List[Dict[str, object]] = []
    with open(args.trajectories, "r", encoding="utf-8") as f:
        for line in f:
            trajectories.append(json.loads(line))

    test_eps = np.where(ep_split == 2)[0]
    if test_eps.size > args.max_test_episodes:
        test_eps = rng.choice(test_eps, size=args.max_test_episodes, replace=False)

    mlp_success = 0
    mlp_collisions = 0
    mlp_timeouts = 0
    mlp_opt_ratios: List[float] = []
    mlp_decision_times: List[float] = []

    astar_success = 0
    astar_opt_ratios: List[float] = []
    astar_decision_times: List[float] = []

    patch_size = int(ckpt["patch_size"])

    for ep in test_eps:
        grid = ep_grids[ep]
        start = tuple(map(int, ep_starts[ep]))
        goal = tuple(map(int, ep_goals[ep]))

        # A* baseline timing and optimal path.
        t0 = time.perf_counter()
        p_astar = astar_path(grid, start, goal)
        t_astar = time.perf_counter() - t0
        if p_astar is not None and len(p_astar) >= 2:
            astar_success += 1
            astar_opt_ratios.append(1.0)
            astar_decision_times.append(t_astar / max(len(p_astar) - 1, 1))

        # Optimal cost from stored expert trajectory.
        p_opt = [tuple(pt) for pt in trajectories[int(ep)]["path"]]
        opt_cost = path_cost(p_opt)

        roll = rollout_policy(
            model,
            grid,
            start,
            goal,
            patch_size=patch_size,
            device=device,
            max_revisit=args.max_revisit,
        )
        mlp_decision_times.append(float(roll["mean_decision_time_s"]))

        if bool(roll["success"]):
            mlp_success += 1
            ratio = float(roll["path_cost"]) / max(opt_cost, 1e-8)
            mlp_opt_ratios.append(ratio)
        else:
            if bool(roll["collision"]):
                mlp_collisions += 1
            if bool(roll["timeout"]):
                mlp_timeouts += 1

    n_eps = max(len(test_eps), 1)
    kpis = {
        "astar": {
            "success_rate": astar_success / n_eps,
            "optimality_ratio": float(np.mean(astar_opt_ratios)) if astar_opt_ratios else float("nan"),
            "decision_time_ms": float(np.mean(astar_decision_times) * 1000.0) if astar_decision_times else float("nan"),
        },
        "mlp": {
            "success_rate": mlp_success / n_eps,
            "optimality_ratio": float(np.mean(mlp_opt_ratios)) if mlp_opt_ratios else float("nan"),
            "decision_time_ms": float(np.mean(mlp_decision_times) * 1000.0) if mlp_decision_times else float("nan"),
        },
    }

    additional = {
        "collision_rate": mlp_collisions / n_eps,
        "timeout_rate": mlp_timeouts / n_eps,
        "test_action_acc": test_metrics["action_accuracy"],
        "test_cost_rmse": test_metrics["cost_rmse"],
    }

    summary = {
        "device": str(device),
        "epochs": args.epochs,
        "action_loss": args.action_loss,
        "best_val_total_loss": best_val,
        "val_step_metrics": val_metrics,
        "test_step_metrics": test_metrics,
        "kpis": kpis,
        "additional_metrics": additional,
        "num_test_episodes_eval": int(len(test_eps)),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, art_dir / "training_curves.png")
    plot_kpi_comparison(kpis, art_dir / "kpi_comparison_astar_vs_torch.png")
    plot_additional_metrics(additional, mlp_opt_ratios, art_dir / "additional_metrics.png")

    print("\nTraining + evaluation complete.")
    print(json.dumps(summary, indent=2))
    print(f"Saved model: {best_path}")
    print(f"Artifacts: {art_dir}")


if __name__ == "__main__":
    main()
