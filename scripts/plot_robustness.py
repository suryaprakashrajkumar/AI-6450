from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _plot_stress(stress_rows: list[dict[str, str]], out_dir: Path) -> None:
    scenarios = sorted({r["scenario"] for r in stress_rows if r["scenario"] != "clean"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for scenario in scenarios:
        rows = [r for r in stress_rows if r["scenario"] == scenario]
        rows = sorted(rows, key=lambda r: int(float(r["severity"])))
        x = [int(float(r["severity"])) for r in rows]
        y_valid = [_to_float(r["path_validity"]) for r in rows]
        y_gap = [_to_float(r["optimality_gap_mean"]) for r in rows]
        axes[0].plot(x, y_valid, marker="o", label=scenario)
        axes[1].plot(x, y_gap, marker="o", label=scenario)

    axes[0].set_title("Stress: Path Validity vs Severity")
    axes[0].set_xlabel("severity")
    axes[0].set_ylabel("path_validity")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Stress: Optimality Gap vs Severity")
    axes[1].set_xlabel("severity")
    axes[1].set_ylabel("optimality_gap_mean")
    axes[1].grid(alpha=0.3)

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "stress_curves.png", dpi=170)
    plt.close(fig)


def _plot_adversarial(adv_rows: list[dict[str, str]], out_dir: Path) -> None:
    rows = [r for r in adv_rows if r.get("strength") is not None]
    rows = sorted(rows, key=lambda r: int(float(r["strength"])))

    x = [int(float(r["strength"])) for r in rows]
    y_valid = [_to_float(r["path_validity"]) for r in rows]
    y_exact = [_to_float(r["exact_match"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y_valid, marker="o", label="path_validity")
    ax.plot(x, y_exact, marker="o", label="exact_match")
    ax.set_title("Adversarial Robustness vs Attack Strength")
    ax.set_xlabel("attack strength")
    ax.set_ylabel("metric")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "adversarial_curves.png", dpi=170)
    plt.close(fig)


def _write_summary_table(stress_rows: list[dict[str, str]], adv_rows: list[dict[str, str]], out_dir: Path) -> None:
    table_path = out_dir / "robustness_table.csv"
    fields = [
        "suite",
        "scenario",
        "severity_or_strength",
        "path_validity",
        "exact_match",
        "optimality_gap_mean",
        "policy_ms_per_query",
    ]

    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in stress_rows:
            writer.writerow(
                {
                    "suite": "stress",
                    "scenario": r.get("scenario", ""),
                    "severity_or_strength": r.get("severity", ""),
                    "path_validity": r.get("path_validity", ""),
                    "exact_match": r.get("exact_match", ""),
                    "optimality_gap_mean": r.get("optimality_gap_mean", ""),
                    "policy_ms_per_query": r.get("policy_ms_per_query", ""),
                }
            )
        for r in adv_rows:
            writer.writerow(
                {
                    "suite": "adversarial",
                    "scenario": r.get("scenario", ""),
                    "severity_or_strength": r.get("strength", ""),
                    "path_validity": r.get("path_validity", ""),
                    "exact_match": r.get("exact_match", ""),
                    "optimality_gap_mean": r.get("optimality_gap_mean", ""),
                    "policy_ms_per_query": r.get("policy_ms_per_query", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress_csv", type=str, default="logs/stress_tests/stress_results.csv")
    parser.add_argument("--adv_csv", type=str, default="logs/adversarial/adversarial_results.csv")
    parser.add_argument("--out_dir", type=str, default="logs/robustness_plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stress_rows = _read_csv(args.stress_csv)
    adv_rows = _read_csv(args.adv_csv)

    _plot_stress(stress_rows, out_dir)
    _plot_adversarial(adv_rows, out_dir)
    _write_summary_table(stress_rows, adv_rows, out_dir)

    print(f"Saved robustness plots and table to {out_dir}")


if __name__ == "__main__":
    main()
