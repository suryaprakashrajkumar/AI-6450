from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import torch

from src.config import load_config
from src.model import build_model
from src.utils import get_torch_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_improved.json")
    parser.add_argument("--model", type=str, default="models/final/imitation_policy_improved.pth")
    parser.add_argument("--test_steps", type=str, default="data/processed/test_samples.npz")
    parser.add_argument("--out_dir", type=str, default="logs/calibration")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--sample_limit", type=int, default=20000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    device = get_torch_device()
    ckpt = torch.load(args.model, map_location=device)
    model = build_model(
        model_type=ckpt.get("model_type", cfg.model_type),
        hidden_dim=ckpt.get("hidden_dim", cfg.hidden_dim),
        n_actions=ckpt.get("n_actions", cfg.n_actions),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    payload = np.load(args.test_steps)
    grids = payload["grids"].astype(np.float32)
    positions = payload["positions"].astype(np.float32)
    goals = payload["goals"].astype(np.float32)
    actions = payload["actions"].astype(np.int64)

    n = min(int(args.sample_limit), len(actions))
    grids = grids[:n]
    positions = positions[:n]
    goals = goals[:n]
    actions = actions[:n]

    confidences = np.zeros((n,), dtype=np.float64)
    correct = np.zeros((n,), dtype=np.float64)

    with torch.no_grad():
        for i in range(n):
            grid_t = torch.from_numpy(grids[i]).unsqueeze(0).unsqueeze(0).to(device)
            pos_t = torch.tensor([[positions[i][0] / 9.0, positions[i][1] / 9.0]], dtype=torch.float32, device=device)
            goal_t = torch.tensor([[goals[i][0] / 9.0, goals[i][1] / 9.0]], dtype=torch.float32, device=device)
            logits = model(grid_t, pos_t, goal_t)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))
            confidences[i] = conf
            correct[i] = 1.0 if pred == int(actions[i]) else 0.0

    avg_conf = float(np.mean(confidences))
    acc = float(np.mean(correct))

    edges = np.linspace(0.0, 1.0, num=args.bins + 1)
    ece = 0.0
    rows: list[dict[str, float]] = []
    for b in range(args.bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidences >= lo) & (confidences < hi if b < args.bins - 1 else confidences <= hi)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin": float(b),
                    "lo": float(lo),
                    "hi": float(hi),
                    "count": 0.0,
                    "bin_accuracy": float("nan"),
                    "bin_confidence": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        gap = abs(bin_acc - bin_conf)
        ece += (count / n) * gap
        rows.append(
            {
                "bin": float(b),
                "lo": float(lo),
                "hi": float(hi),
                "count": float(count),
                "bin_accuracy": bin_acc,
                "bin_confidence": bin_conf,
                "gap": float(gap),
            }
        )

    # Save bin table
    csv_path = out_dir / "calibration_bins.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin", "lo", "hi", "count", "bin_accuracy", "bin_confidence", "gap"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Confidence histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(confidences, bins=args.bins, color="#1D4ED8", alpha=0.85)
    ax.set_title("Confidence Histogram")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_histogram.png", dpi=170)
    plt.close(fig)

    # Reliability diagram
    bin_centers = []
    bin_accs = []
    for row in rows:
        if np.isnan(row["bin_accuracy"]):
            continue
        bin_centers.append((row["lo"] + row["hi"]) / 2.0)
        bin_accs.append(row["bin_accuracy"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(bin_centers, bin_accs, marker="o", color="#DC2626", label="model")
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reliability_diagram.png", dpi=170)
    plt.close(fig)

    summary = {
        "samples": int(n),
        "accuracy": acc,
        "mean_confidence": avg_conf,
        "ece": float(ece),
    }
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
