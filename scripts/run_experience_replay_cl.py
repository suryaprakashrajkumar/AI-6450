from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from src.environment import AStarPlanner, Point, delta_to_action
from src.eval import evaluate_model
from src.model import build_model
from src.utils import get_torch_device, set_seed


class ArrayImitationDataset(Dataset):
    def __init__(self, grids: np.ndarray, positions: np.ndarray, goals: np.ndarray, actions: np.ndarray):
        self.grids = grids.astype(np.float32)
        self.positions = positions.astype(np.float32)
        self.goals = goals.astype(np.float32)
        self.actions = actions.astype(np.int64)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        grid = torch.from_numpy(self.grids[idx]).unsqueeze(0)
        pos = torch.from_numpy(self.positions[idx]) / 9.0
        goal = torch.from_numpy(self.goals[idx]) / 9.0
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        return grid, pos, goal, action


def _build_weighted_loss(actions: np.ndarray, n_actions: int, label_smoothing: float) -> nn.CrossEntropyLoss:
    action_t = torch.tensor(actions, dtype=torch.int64)
    counts = torch.bincount(action_t, minlength=n_actions).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * n_actions)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total = 0
    correct = 0
    for grid, pos, goal, action in loader:
        grid = grid.to(device)
        pos = pos.to(device)
        goal = goal.to(device)
        action = action.to(device)

        logits = model(grid, pos, goal)
        loss = criterion(logits, action)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * action.size(0)
        total += action.size(0)
        correct += (logits.argmax(dim=1) == action).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)


def _metric_score(metrics: dict[str, float | str], metric: str) -> float:
    higher_is_better = {
        "path_validity",
        "exact_match",
        "step_accuracy",
        "success_under_1_5x",
    }
    lower_is_better = {
        "optimality_gap_mean",
        "optimality_gap_p95",
        "path_length_ratio_mean",
        "policy_ms_per_query",
        "astar_ms_per_query",
    }
    val = float(metrics[metric])
    if metric in higher_is_better:
        return val
    if metric in lower_is_better:
        return -val
    raise ValueError(f"Unsupported selection_metric '{metric}'.")


def _save_checkpoint(model: nn.Module, model_type: str, hidden_dim: int, n_actions: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "hidden_dim": int(hidden_dim),
            "n_actions": int(n_actions),
            "model_type": model_type,
        },
        output_path,
    )


def _collect_expert_trajectory_samples(
    planner: AStarPlanner,
    grid: np.ndarray,
    start: Point,
    goal: Point,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    path = planner.solve(grid, start, goal)
    if path is None or len(path) < 2:
        return [], [], [], []

    out_grids: list[np.ndarray] = []
    out_positions: list[np.ndarray] = []
    out_goals: list[np.ndarray] = []
    out_actions: list[int] = []

    for i in range(len(path) - 1):
        cur = path[i]
        nxt = path[i + 1]
        delta = (nxt.row - cur.row, nxt.col - cur.col)
        action = delta_to_action(delta)
        out_grids.append(grid.astype(np.float32))
        out_positions.append(np.array([cur.row, cur.col], dtype=np.float32))
        out_goals.append(np.array([goal.row, goal.col], dtype=np.float32))
        out_actions.append(int(action))

    return out_grids, out_positions, out_goals, out_actions


def _select_drift_task_indices(
    densities: np.ndarray,
    tasks_per_update: int,
    progress: float,
    start_quantile: float,
    end_quantile: float,
    rng: np.random.Generator,
) -> np.ndarray:
    q = float(start_quantile + (end_quantile - start_quantile) * progress)
    q = float(np.clip(q, 0.0, 1.0))
    threshold = float(np.quantile(densities, q))
    candidates = np.where(densities >= threshold)[0]
    if len(candidates) == 0:
        candidates = np.arange(len(densities))

    n = min(int(tasks_per_update), len(candidates))
    return rng.choice(candidates, size=n, replace=False)


def _reservoir_append(
    old_grids: np.ndarray,
    old_positions: np.ndarray,
    old_goals: np.ndarray,
    old_actions: np.ndarray,
    new_grids: np.ndarray,
    new_positions: np.ndarray,
    new_goals: np.ndarray,
    new_actions: np.ndarray,
    capacity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(new_actions) == 0:
        return old_grids, old_positions, old_goals, old_actions

    if len(old_actions) == 0:
        merged_grids = new_grids
        merged_positions = new_positions
        merged_goals = new_goals
        merged_actions = new_actions
    else:
        merged_grids = np.concatenate([old_grids, new_grids], axis=0)
        merged_positions = np.concatenate([old_positions, new_positions], axis=0)
        merged_goals = np.concatenate([old_goals, new_goals], axis=0)
        merged_actions = np.concatenate([old_actions, new_actions], axis=0)

    if len(merged_actions) <= capacity:
        return merged_grids, merged_positions, merged_goals, merged_actions

    idx = rng.choice(len(merged_actions), size=capacity, replace=False)
    return merged_grids[idx], merged_positions[idx], merged_goals[idx], merged_actions[idx]


def _compose_training_arrays(
    base_grids: np.ndarray,
    base_positions: np.ndarray,
    base_goals: np.ndarray,
    base_actions: np.ndarray,
    replay_grids: np.ndarray,
    replay_positions: np.ndarray,
    replay_goals: np.ndarray,
    replay_actions: np.ndarray,
    replay_fraction: float,
    replay_min_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(replay_actions) == 0:
        return base_grids, base_positions, base_goals, base_actions

    replay_fraction = float(np.clip(replay_fraction, 0.0, 1.0))
    if replay_fraction >= 1.0:
        n_rep_target = len(replay_actions)
    elif replay_fraction <= 0.0:
        n_rep_target = 0
    else:
        n_rep_target = int((replay_fraction / max(1e-8, 1.0 - replay_fraction)) * len(base_actions))

    n_rep = int(min(len(replay_actions), max(int(replay_min_samples), n_rep_target)))
    if n_rep > 0:
        rep_idx = rng.choice(len(replay_actions), size=n_rep, replace=False)
        rgr = replay_grids[rep_idx]
        rpo = replay_positions[rep_idx]
        rgo = replay_goals[rep_idx]
        rac = replay_actions[rep_idx]
    else:
        rgr = np.zeros((0, 10, 10), dtype=np.float32)
        rpo = np.zeros((0, 2), dtype=np.float32)
        rgo = np.zeros((0, 2), dtype=np.float32)
        rac = np.zeros((0,), dtype=np.int64)

    tr_grids = np.concatenate([base_grids, rgr], axis=0)
    tr_positions = np.concatenate([base_positions, rpo], axis=0)
    tr_goals = np.concatenate([base_goals, rgo], axis=0)
    tr_actions = np.concatenate([base_actions, rac], axis=0)
    return tr_grids, tr_positions, tr_goals, tr_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_scaleup.json")
    parser.add_argument("--base_model", type=str, default="models/final/imitation_policy_scaleup.pth")
    parser.add_argument("--train_rollout", type=str, default="data/processed_scaleup/train_rollout.npz")
    parser.add_argument("--train_steps", type=str, default="data/processed_scaleup/train_samples.npz")
    parser.add_argument("--val_steps", type=str, default="data/processed_scaleup/val_samples.npz")
    parser.add_argument("--test_rollout", type=str, default="data/processed_scaleup/test_rollout.npz")
    parser.add_argument("--test_steps", type=str, default="data/processed_scaleup/test_samples.npz")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--tasks_per_update", type=int, default=256)
    parser.add_argument("--epochs_per_update", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--replay_capacity", type=int, default=50000)
    parser.add_argument("--replay_fraction", type=float, default=0.35)
    parser.add_argument("--replay_min_samples", type=int, default=12000)
    parser.add_argument("--drift_start_quantile", type=float, default=0.60)
    parser.add_argument("--drift_end_quantile", type=float, default=0.90)
    parser.add_argument("--selection_metric", type=str, default="exact_match")
    parser.add_argument("--selection_rollout_limit", type=int, default=800)
    parser.add_argument("--selection_step_limit", type=int, default=8000)
    parser.add_argument("--out_dir", type=str, default="logs/replay_cl")
    parser.add_argument("--output_model", type=str, default="models/final/imitation_policy_scaleup_replay_cl.pth")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_torch_device()
    ckpt = torch.load(args.base_model, map_location=device)
    model_type = str(ckpt.get("model_type", cfg.model_type))
    hidden_dim = int(ckpt.get("hidden_dim", cfg.hidden_dim))
    n_actions = int(ckpt.get("n_actions", cfg.n_actions))

    model = build_model(model_type=model_type, hidden_dim=hidden_dim, n_actions=n_actions)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    train_rollout_payload = np.load(args.train_rollout)
    roll_grids = train_rollout_payload["grids"]
    roll_starts = train_rollout_payload["starts"]
    roll_goals = train_rollout_payload["goals_full"]
    map_densities = roll_grids.mean(axis=(1, 2)).astype(np.float32)

    train_payload = np.load(args.train_steps)
    base_grids = train_payload["grids"].astype(np.float32)
    base_positions = train_payload["positions"].astype(np.float32)
    base_goals = train_payload["goals"].astype(np.float32)
    base_actions = train_payload["actions"].astype(np.int64)

    val_payload = np.load(args.val_steps)
    val_ds = ArrayImitationDataset(
        val_payload["grids"],
        val_payload["positions"],
        val_payload["goals"],
        val_payload["actions"],
    )

    replay_grids = np.zeros((0, 10, 10), dtype=np.float32)
    replay_positions = np.zeros((0, 2), dtype=np.float32)
    replay_goals = np.zeros((0, 2), dtype=np.float32)
    replay_actions = np.zeros((0,), dtype=np.int64)

    planner = AStarPlanner()
    rng = np.random.default_rng(cfg.seed)

    lr = float(args.lr) if args.lr is not None else float(cfg.lr)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(cfg.weight_decay))

    history: list[dict[str, float]] = []
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_iter = 0
    best_metrics: dict[str, float | str] | None = None

    cfg_no_fallback = asdict(cfg)
    cfg_no_fallback["enable_astar_fallback"] = False
    cfg_no_fallback["enable_confidence_fallback"] = False
    no_fallback_path = out_dir / "config_no_fallback.json"
    no_fallback_path.write_text(json.dumps(cfg_no_fallback, indent=2), encoding="utf-8")

    for it in range(1, int(args.updates) + 1):
        iter_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        progress = 0.0 if int(args.updates) <= 1 else (it - 1) / (int(args.updates) - 1)
        task_indices = _select_drift_task_indices(
            densities=map_densities,
            tasks_per_update=int(args.tasks_per_update),
            progress=progress,
            start_quantile=float(args.drift_start_quantile),
            end_quantile=float(args.drift_end_quantile),
            rng=rng,
        )

        batch_grids: list[np.ndarray] = []
        batch_positions: list[np.ndarray] = []
        batch_goals: list[np.ndarray] = []
        batch_actions: list[int] = []

        for idx in task_indices:
            grid = roll_grids[int(idx)]
            start = Point(int(roll_starts[int(idx)][0]), int(roll_starts[int(idx)][1]))
            goal = Point(int(roll_goals[int(idx)][0]), int(roll_goals[int(idx)][1]))
            g, p, go, a = _collect_expert_trajectory_samples(planner, grid, start, goal)
            batch_grids.extend(g)
            batch_positions.extend(p)
            batch_goals.extend(go)
            batch_actions.extend(a)

        if len(batch_actions) == 0:
            print(f"iter={it:02d} no stream samples collected; skipping update")
            continue

        new_grids = np.stack(batch_grids, axis=0).astype(np.float32)
        new_positions = np.stack(batch_positions, axis=0).astype(np.float32)
        new_goals = np.stack(batch_goals, axis=0).astype(np.float32)
        new_actions = np.array(batch_actions, dtype=np.int64)

        replay_grids, replay_positions, replay_goals, replay_actions = _reservoir_append(
            old_grids=replay_grids,
            old_positions=replay_positions,
            old_goals=replay_goals,
            old_actions=replay_actions,
            new_grids=new_grids,
            new_positions=new_positions,
            new_goals=new_goals,
            new_actions=new_actions,
            capacity=int(args.replay_capacity),
            rng=rng,
        )

        tr_grids, tr_positions, tr_goals, tr_actions = _compose_training_arrays(
            base_grids=base_grids,
            base_positions=base_positions,
            base_goals=base_goals,
            base_actions=base_actions,
            replay_grids=replay_grids,
            replay_positions=replay_positions,
            replay_goals=replay_goals,
            replay_actions=replay_actions,
            replay_fraction=float(args.replay_fraction),
            replay_min_samples=int(args.replay_min_samples),
            rng=rng,
        )

        train_ds = ArrayImitationDataset(tr_grids, tr_positions, tr_goals, tr_actions)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        criterion = _build_weighted_loss(
            tr_actions,
            n_actions=n_actions,
            label_smoothing=float(cfg.label_smoothing),
        ).to(device)

        last_train_loss = 0.0
        last_train_acc = 0.0
        last_val_loss = 0.0
        last_val_acc = 0.0
        for _ in range(int(args.epochs_per_update)):
            last_train_loss, last_train_acc = _epoch(model, train_loader, criterion, optimizer, device)
            last_val_loss, last_val_acc = _epoch(model, val_loader, criterion, None, device)

        iter_model_path = out_dir / f"iter_{it:02d}_model.pth"
        _save_checkpoint(model, model_type=model_type, hidden_dim=hidden_dim, n_actions=n_actions, output_path=iter_model_path)

        sel_metrics = evaluate_model(
            rollout_npz_path=args.test_rollout,
            step_npz_path=args.test_steps,
            model_path=str(iter_model_path),
            config_path=args.config,
            metrics_output=None,
            compute_step_accuracy=True,
            rollout_limit=int(args.selection_rollout_limit),
            step_limit=int(args.selection_step_limit),
        )
        sel_score = _metric_score(sel_metrics, str(args.selection_metric))
        if sel_score > best_score:
            best_score = sel_score
            best_iter = it
            best_metrics = sel_metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        iter_duration_s = float(time.perf_counter() - iter_start)
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            if torch.cuda.is_available()
            else 0.0
        )

        print(
            f"iter={it:02d} q={float(args.drift_start_quantile + (args.drift_end_quantile - args.drift_start_quantile) * progress):.3f} "
            f"new_samples={len(new_actions)} replay_pool={len(replay_actions)} train_subset={len(tr_actions)} "
            f"train_loss={last_train_loss:.4f} train_acc={last_train_acc:.4f} "
            f"val_loss={last_val_loss:.4f} val_acc={last_val_acc:.4f} "
            f"sel_{args.selection_metric}={float(sel_metrics[str(args.selection_metric)]):.4f}"
        )

        history.append(
            {
                "iteration": float(it),
                "drift_quantile": float(
                    args.drift_start_quantile + (args.drift_end_quantile - args.drift_start_quantile) * progress
                ),
                "new_samples": float(len(new_actions)),
                "replay_pool_samples": float(len(replay_actions)),
                "train_subset_samples": float(len(tr_actions)),
                "train_loss": float(last_train_loss),
                "train_acc": float(last_train_acc),
                "val_loss": float(last_val_loss),
                "val_acc": float(last_val_acc),
                "selection_score": float(sel_score),
                "selection_metric_value": float(sel_metrics[str(args.selection_metric)]),
                "iter_duration_s": iter_duration_s,
                "peak_vram_mb": peak_vram_mb,
            }
        )

    model.load_state_dict(best_state)
    output_model_path = Path(args.output_model)
    _save_checkpoint(model, model_type=model_type, hidden_dim=hidden_dim, n_actions=n_actions, output_path=output_model_path)

    baseline_metrics = evaluate_model(
        rollout_npz_path=args.test_rollout,
        step_npz_path=args.test_steps,
        model_path=args.base_model,
        config_path=args.config,
        metrics_output=str(out_dir / "baseline_eval_kpi.json"),
        compute_step_accuracy=True,
    )

    replay_metrics = evaluate_model(
        rollout_npz_path=args.test_rollout,
        step_npz_path=args.test_steps,
        model_path=str(output_model_path),
        config_path=args.config,
        metrics_output=str(out_dir / "replay_eval_kpi.json"),
        compute_step_accuracy=True,
    )

    baseline_no_fallback_metrics = evaluate_model(
        rollout_npz_path=args.test_rollout,
        step_npz_path=args.test_steps,
        model_path=args.base_model,
        config_path=str(no_fallback_path),
        metrics_output=str(out_dir / "baseline_eval_kpi_no_fallback.json"),
        compute_step_accuracy=True,
    )

    replay_no_fallback_metrics = evaluate_model(
        rollout_npz_path=args.test_rollout,
        step_npz_path=args.test_steps,
        model_path=str(output_model_path),
        config_path=str(no_fallback_path),
        metrics_output=str(out_dir / "replay_eval_kpi_no_fallback.json"),
        compute_step_accuracy=True,
    )

    compare_keys = [
        "path_validity",
        "exact_match",
        "step_accuracy",
        "optimality_gap_mean",
        "optimality_gap_p95",
        "success_under_1_5x",
        "path_length_ratio_mean",
        "policy_ms_per_query",
        "astar_ms_per_query",
    ]
    deltas: dict[str, float] = {}
    for key in compare_keys:
        deltas[f"delta_{key}"] = float(replay_metrics[key]) - float(baseline_metrics[key])

    deltas_no_fallback: dict[str, float] = {}
    for key in compare_keys:
        deltas_no_fallback[f"delta_{key}"] = float(replay_no_fallback_metrics[key]) - float(
            baseline_no_fallback_metrics[key]
        )

    summary = {
        "config": asdict(cfg),
        "updates": int(args.updates),
        "tasks_per_update": int(args.tasks_per_update),
        "epochs_per_update": int(args.epochs_per_update),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "replay_capacity": int(args.replay_capacity),
        "replay_fraction": float(args.replay_fraction),
        "replay_min_samples": int(args.replay_min_samples),
        "drift_start_quantile": float(args.drift_start_quantile),
        "drift_end_quantile": float(args.drift_end_quantile),
        "selection_metric": str(args.selection_metric),
        "best_iteration": int(best_iter),
        "best_selection_score": float(best_score),
        "best_selection_metrics": best_metrics,
        "base_model": str(args.base_model),
        "replay_model": str(output_model_path),
        "history": history,
        "baseline": baseline_metrics,
        "replay": replay_metrics,
        "baseline_no_fallback": baseline_no_fallback_metrics,
        "replay_no_fallback": replay_no_fallback_metrics,
        "deltas": deltas,
        "deltas_no_fallback": deltas_no_fallback,
    }
    (out_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  replay model: {output_model_path}")
    print(f"  baseline kpi: {out_dir / 'baseline_eval_kpi.json'}")
    print(f"  replay kpi:   {out_dir / 'replay_eval_kpi.json'}")
    print(f"  baseline no-fallback kpi: {out_dir / 'baseline_eval_kpi_no_fallback.json'}")
    print(f"  replay no-fallback kpi:   {out_dir / 'replay_eval_kpi_no_fallback.json'}")
    print(f"  summary:      {out_dir / 'replay_summary.json'}")


if __name__ == "__main__":
    main()
