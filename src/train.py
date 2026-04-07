from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import ImitationDataset, run_quality_checks
from src.model import build_model
from src.utils import get_torch_device, set_seed


def _build_weighted_loss(actions: torch.Tensor, n_actions: int = 8, label_smoothing: float = 0.0) -> nn.CrossEntropyLoss:
    counts = torch.bincount(actions, minlength=n_actions).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * n_actions)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def _plot_history(history: list[dict[str, float]], output_png: str) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    lrs = [row["lr"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, train_loss, label="train_loss", color="#1D4ED8")
    axes[0].plot(epochs, val_loss, label="val_loss", color="#DC2626")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_acc", color="#047857")
    axes[1].plot(epochs, val_acc, label="val_acc", color="#B45309")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, lrs, label="lr", color="#7C3AED")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device):
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


def train_model(
    config_path: str | None,
    train_npz: str,
    val_npz: str,
    output_path: str,
    model_type: str | None = None,
    hidden_dim: int | None = None,
    metrics_output: str | None = None,
    history_output: str | None = None,
    history_plot_output: str | None = None,
) -> dict[str, float | str]:
    config = load_config(config_path)
    set_seed(config.seed)

    selected_model_type = model_type if model_type is not None else config.model_type
    selected_hidden_dim = hidden_dim if hidden_dim is not None else config.hidden_dim

    train_report = run_quality_checks(train_npz, config.qa_action_imbalance_ratio_threshold)
    val_report = run_quality_checks(val_npz, config.qa_action_imbalance_ratio_threshold)
    if not train_report.passed or not val_report.passed:
        raise RuntimeError("Dataset QA failed. Run scripts/quality_check.py for details.")

    train_ds = ImitationDataset(train_npz)
    val_ds = ImitationDataset(val_npz)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = build_model(selected_model_type, hidden_dim=selected_hidden_dim, n_actions=config.n_actions)
    device = get_torch_device()
    model.to(device)

    action_tensor = torch.tensor(train_ds.actions, dtype=torch.int64)
    criterion = _build_weighted_loss(
        action_tensor,
        n_actions=config.n_actions,
        label_smoothing=float(config.label_smoothing),
    )
    criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    patience = 0
    completed_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _epoch(model, val_loader, criterion, None, device)
        completed_epochs = epoch
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        scheduler.step(val_acc)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": current_lr,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        best_state = model.state_dict()

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "hidden_dim": selected_hidden_dim,
            "n_actions": config.n_actions,
            "model_type": selected_model_type,
        },
        output_path_obj,
    )
    print(f"Saved best model to {output_path_obj}")

    metrics: dict[str, float | str] = {
        "model_type": selected_model_type,
        "hidden_dim": float(selected_hidden_dim),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "epochs_completed": float(completed_epochs),
        "train_samples": float(len(train_ds)),
        "val_samples": float(len(val_ds)),
    }

    if metrics_output is not None:
        metrics_path = Path(metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if history_output is not None:
        hist_path = Path(history_output)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with hist_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
            writer.writeheader()
            writer.writerows(history)

    if history_plot_output is not None and history:
        _plot_history(history, history_plot_output)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train", type=str, default="data/processed/train_samples.npz")
    parser.add_argument("--val", type=str, default="data/processed/val_samples.npz")
    parser.add_argument("--output", type=str, default="models/final/imitation_policy.pth")
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--metrics_output", type=str, default=None)
    parser.add_argument("--history_output", type=str, default=None)
    parser.add_argument("--history_plot_output", type=str, default=None)
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        train_npz=args.train,
        val_npz=args.val,
        output_path=args.output,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        metrics_output=args.metrics_output,
        history_output=args.history_output,
        history_plot_output=args.history_plot_output,
    )


if __name__ == "__main__":
    main()
