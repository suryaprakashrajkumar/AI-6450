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
from src.environment import AStarPlanner, Point, action_to_delta, delta_to_action
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


def _policy_action_and_confidence(
    model: nn.Module,
    device: torch.device,
    grid: np.ndarray,
    cur: Point,
    goal: Point,
    visited: set[tuple[int, int]],
) -> tuple[int | None, float]:
    grid_t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    pos_t = torch.tensor([[cur.row / 9.0, cur.col / 9.0]], dtype=torch.float32, device=device)
    goal_t = torch.tensor([[goal.row / 9.0, goal.col / 9.0]], dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(grid_t, pos_t, goal_t)
        probs = torch.softmax(logits, dim=1)
        max_conf = float(torch.max(probs).item())
        ranked = torch.argsort(logits, descending=True).squeeze(0).tolist()

    for action in ranked:
        dr, dc = action_to_delta(int(action))
        nr, nc = cur.row + dr, cur.col + dc
        if not (0 <= nr < 10 and 0 <= nc < 10):
            continue
        if grid[nr, nc] == 1:
            continue
        if (nr, nc) in visited and (nr, nc) != (goal.row, goal.col):
            continue
        return int(action), max_conf
    return None, max_conf


def _compose_training_arrays(
    base_grids: np.ndarray,
    base_positions: np.ndarray,
    base_goals: np.ndarray,
    base_actions: np.ndarray,
    dagger_grids: np.ndarray,
    dagger_positions: np.ndarray,
    dagger_goals: np.ndarray,
    dagger_actions: np.ndarray,
    bc_fraction: float,
    bc_min_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(dagger_actions) == 0:
        return base_grids, base_positions, base_goals, base_actions

    bc_fraction = float(np.clip(bc_fraction, 0.0, 1.0))
    if bc_fraction >= 1.0:
        n_bc_target = len(base_actions)
    elif bc_fraction <= 0.0:
        n_bc_target = 0
    else:
        n_bc_target = int((bc_fraction / max(1e-8, 1.0 - bc_fraction)) * len(dagger_actions))

    n_bc = int(min(len(base_actions), max(int(bc_min_samples), n_bc_target)))
    if n_bc > 0:
        base_idx = rng.choice(len(base_actions), size=n_bc, replace=False)
        bgr = base_grids[base_idx]
        bpo = base_positions[base_idx]
        bgo = base_goals[base_idx]
        bac = base_actions[base_idx]
    else:
        bgr = np.zeros((0, 10, 10), dtype=np.float32)
        bpo = np.zeros((0, 2), dtype=np.float32)
        bgo = np.zeros((0, 2), dtype=np.float32)
        bac = np.zeros((0,), dtype=np.int64)

    tr_grids = np.concatenate([bgr, dagger_grids], axis=0)
    tr_positions = np.concatenate([bpo, dagger_positions], axis=0)
    tr_goals = np.concatenate([bgo, dagger_goals], axis=0)
    tr_actions = np.concatenate([bac, dagger_actions], axis=0)
    return tr_grids, tr_positions, tr_goals, tr_actions


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


def collect_dagger_samples(
    model: nn.Module,
    planner: AStarPlanner,
    grids: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    task_indices: np.ndarray,
    step_cap: int,
    device: torch.device,
    beta: float,
    conf_threshold: float,
    high_conf_keep_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_grids: list[np.ndarray] = []
    out_positions: list[np.ndarray] = []
    out_goals: list[np.ndarray] = []
    out_actions: list[int] = []

    model.eval()

    for idx in task_indices:
        grid = grids[int(idx)]
        cur = Point(int(starts[int(idx)][0]), int(starts[int(idx)][1]))
        goal = Point(int(goals[int(idx)][0]), int(goals[int(idx)][1]))
        visited: set[tuple[int, int]] = {(cur.row, cur.col)}

        for _ in range(step_cap):
            if cur == goal:
                break

            expert_path = planner.solve(grid, cur, goal)
            if expert_path is None or len(expert_path) < 2:
                break

            delta = (expert_path[1].row - expert_path[0].row, expert_path[1].col - expert_path[0].col)
            expert_action = delta_to_action(delta)

            policy_action, max_conf = _policy_action_and_confidence(model, device, grid, cur, goal, visited)
            keep_label = (max_conf < conf_threshold) or (rng.random() < high_conf_keep_prob)
            if keep_label:
                out_grids.append(grid.astype(np.float32))
                out_positions.append(np.array([cur.row, cur.col], dtype=np.float32))
                out_goals.append(np.array([goal.row, goal.col], dtype=np.float32))
                out_actions.append(int(expert_action))

            # Beta-mixing: use expert action early, decay toward policy actions.
            if policy_action is None:
                chosen = expert_action
            else:
                chosen = expert_action if (rng.random() < beta) else policy_action

            dr, dc = action_to_delta(int(chosen))
            cur = Point(cur.row + dr, cur.col + dc)
            visited.add((cur.row, cur.col))

    if not out_actions:
        return (
            np.zeros((0, 10, 10), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(out_grids, axis=0),
        np.stack(out_positions, axis=0),
        np.stack(out_goals, axis=0),
        np.array(out_actions, dtype=np.int64),
    )


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_scaleup.json")
    parser.add_argument("--base_model", type=str, default="models/final/imitation_policy_scaleup.pth")
    parser.add_argument("--train_rollout", type=str, default="data/processed_scaleup/train_rollout.npz")
    parser.add_argument("--train_steps", type=str, default="data/processed_scaleup/train_samples.npz")
    parser.add_argument("--val_steps", type=str, default="data/processed_scaleup/val_samples.npz")
    parser.add_argument("--test_rollout", type=str, default="data/processed_scaleup/test_rollout.npz")
    parser.add_argument("--test_steps", type=str, default="data/processed_scaleup/test_samples.npz")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--tasks_per_rollout", type=int, default=128)
    parser.add_argument("--epochs_per_rollout", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--beta_start", type=float, default=0.8)
    parser.add_argument("--beta_end", type=float, default=0.2)
    parser.add_argument("--conf_threshold", type=float, default=0.45)
    parser.add_argument("--high_conf_keep_prob", type=float, default=0.2)
    parser.add_argument("--bc_fraction", type=float, default=0.7)
    parser.add_argument("--bc_min_samples", type=int, default=30000)
    parser.add_argument("--selection_metric", type=str, default="exact_match")
    parser.add_argument("--selection_rollout_limit", type=int, default=600)
    parser.add_argument("--selection_step_limit", type=int, default=6000)
    parser.add_argument("--out_dir", type=str, default="logs/dagger")
    parser.add_argument("--output_model", type=str, default="models/final/imitation_policy_scaleup_dagger.pth")
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

    train_payload = np.load(args.train_steps)
    base_grids = train_payload["grids"].astype(np.float32)
    base_positions = train_payload["positions"].astype(np.float32)
    base_goals = train_payload["goals"].astype(np.float32)
    base_actions = train_payload["actions"].astype(np.int64)

    dag_grids = np.zeros((0, 10, 10), dtype=np.float32)
    dag_positions = np.zeros((0, 2), dtype=np.float32)
    dag_goals = np.zeros((0, 2), dtype=np.float32)
    dag_actions = np.zeros((0,), dtype=np.int64)

    val_payload = np.load(args.val_steps)
    val_ds = ArrayImitationDataset(
        val_payload["grids"],
        val_payload["positions"],
        val_payload["goals"],
        val_payload["actions"],
    )

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

    for it in range(1, int(args.rollouts) + 1):
        iter_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        sample_count = min(int(args.tasks_per_rollout), len(roll_grids))
        task_indices = rng.choice(len(roll_grids), size=sample_count, replace=False)
        progress = 0.0 if int(args.rollouts) <= 1 else (it - 1) / (int(args.rollouts) - 1)
        beta = float(args.beta_start) + (float(args.beta_end) - float(args.beta_start)) * progress

        new_grids, new_positions, new_goals, new_actions = collect_dagger_samples(
            model=model,
            planner=planner,
            grids=roll_grids,
            starts=roll_starts,
            goals=roll_goals,
            task_indices=task_indices,
            step_cap=int(cfg.rollout_step_cap),
            device=device,
            beta=beta,
            conf_threshold=float(args.conf_threshold),
            high_conf_keep_prob=float(args.high_conf_keep_prob),
            rng=rng,
        )

        if len(new_actions) == 0:
            print(f"iter={it:02d} no new samples collected; skipping fine-tuning")
            continue

        dag_grids = np.concatenate([dag_grids, new_grids], axis=0)
        dag_positions = np.concatenate([dag_positions, new_positions], axis=0)
        dag_goals = np.concatenate([dag_goals, new_goals], axis=0)
        dag_actions = np.concatenate([dag_actions, new_actions], axis=0)

        tr_grids, tr_positions, tr_goals, tr_actions = _compose_training_arrays(
            base_grids=base_grids,
            base_positions=base_positions,
            base_goals=base_goals,
            base_actions=base_actions,
            dagger_grids=dag_grids,
            dagger_positions=dag_positions,
            dagger_goals=dag_goals,
            dagger_actions=dag_actions,
            bc_fraction=float(args.bc_fraction),
            bc_min_samples=int(args.bc_min_samples),
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
        for _ in range(int(args.epochs_per_rollout)):
            last_train_loss, last_train_acc = _epoch(model, train_loader, criterion, optimizer, device)
            last_val_loss, last_val_acc = _epoch(model, val_loader, criterion, None, device)

        print(
            f"iter={it:02d} beta={beta:.3f} new_samples={len(new_actions)} dag_pool={len(dag_actions)} "
            f"train_subset={len(tr_actions)} "
            f"train_loss={last_train_loss:.4f} train_acc={last_train_acc:.4f} "
            f"val_loss={last_val_loss:.4f} val_acc={last_val_acc:.4f}"
        )

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

        history.append(
            {
                "iteration": float(it),
                "beta": float(beta),
                "new_samples": float(len(new_actions)),
                "dagger_pool_samples": float(len(dag_actions)),
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

    dagger_metrics = evaluate_model(
        rollout_npz_path=args.test_rollout,
        step_npz_path=args.test_steps,
        model_path=str(output_model_path),
        config_path=args.config,
        metrics_output=str(out_dir / "dagger_eval_kpi.json"),
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

    dagger_no_fallback_metrics = evaluate_model(
        rollout_npz_path=args.test_rollout,
        step_npz_path=args.test_steps,
        model_path=str(output_model_path),
        config_path=str(no_fallback_path),
        metrics_output=str(out_dir / "dagger_eval_kpi_no_fallback.json"),
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
        b = float(baseline_metrics[key])
        d = float(dagger_metrics[key])
        deltas[f"delta_{key}"] = d - b

    deltas_no_fallback: dict[str, float] = {}
    for key in compare_keys:
        b = float(baseline_no_fallback_metrics[key])
        d = float(dagger_no_fallback_metrics[key])
        deltas_no_fallback[f"delta_{key}"] = d - b

    summary = {
        "config": asdict(cfg),
        "rollouts": int(args.rollouts),
        "tasks_per_rollout": int(args.tasks_per_rollout),
        "epochs_per_rollout": int(args.epochs_per_rollout),
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
        "conf_threshold": float(args.conf_threshold),
        "high_conf_keep_prob": float(args.high_conf_keep_prob),
        "bc_fraction": float(args.bc_fraction),
        "bc_min_samples": int(args.bc_min_samples),
        "selection_metric": str(args.selection_metric),
        "best_iteration": int(best_iter),
        "best_selection_score": float(best_score),
        "best_selection_metrics": best_metrics,
        "base_model": str(args.base_model),
        "dagger_model": str(output_model_path),
        "history": history,
        "baseline": baseline_metrics,
        "dagger": dagger_metrics,
        "baseline_no_fallback": baseline_no_fallback_metrics,
        "dagger_no_fallback": dagger_no_fallback_metrics,
        "deltas": deltas,
        "deltas_no_fallback": deltas_no_fallback,
    }
    (out_dir / "dagger_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  dagger model: {output_model_path}")
    print(f"  baseline kpi: {out_dir / 'baseline_eval_kpi.json'}")
    print(f"  dagger kpi:   {out_dir / 'dagger_eval_kpi.json'}")
    print(f"  baseline no-fallback kpi: {out_dir / 'baseline_eval_kpi_no_fallback.json'}")
    print(f"  dagger no-fallback kpi:   {out_dir / 'dagger_eval_kpi_no_fallback.json'}")
    print(f"  summary:      {out_dir / 'dagger_summary.json'}")


if __name__ == "__main__":
    main()
