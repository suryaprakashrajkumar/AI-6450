from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _read_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="logs/ablation/ablation_results.csv")
    parser.add_argument("--out_dir", type=str, default="logs/ablation/charts")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [r["model_type"] for r in rows]

    # KPI chart set
    kpis = [
        ("path_validity", "Path Validity"),
        ("step_accuracy", "Step Accuracy"),
        ("exact_match", "Exact Match"),
        ("success_under_1_5x", "Success <=1.5x"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (k, title) in zip(axes.flatten(), kpis):
        vals = [_to_float(r.get(k, "nan")) for r in rows]
        x = np.arange(len(labels))
        ax.bar(x, vals, color="#2D6A4F")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "kpi_comparison.png", dpi=160)
    plt.close(fig)

    # Rollout timing chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    astar_ms = [_to_float(r.get("astar_ms_per_query", "nan")) for r in rows]
    policy_ms = [_to_float(r.get("policy_ms_per_query", "nan")) for r in rows]
    width = 0.35

    ax.bar(x - width / 2, astar_ms, width=width, label="A* ms/query", color="#1B4332")
    ax.bar(x + width / 2, policy_ms, width=width, label="Policy rollout ms/query", color="#40916C")
    ax.set_title("Rollout Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("milliseconds")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rollout_time_comparison.png", dpi=160)
    plt.close(fig)

    # Optimality gap chart
    fig, ax = plt.subplots(figsize=(10, 5))
    gaps = [_to_float(r.get("optimality_gap_mean", "nan")) for r in rows]
    p95 = [_to_float(r.get("optimality_gap_p95", "nan")) for r in rows]
    ax.plot(labels, gaps, marker="o", label="Mean gap", color="#D97706")
    ax.plot(labels, p95, marker="o", label="P95 gap", color="#B91C1C")
    ax.set_title("Optimality Gap Comparison")
    ax.set_ylabel("gap ratio")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "optimality_gap_comparison.png", dpi=160)
    plt.close(fig)

    print(f"Saved charts to {out_dir}")


if __name__ == "__main__":
    main()
