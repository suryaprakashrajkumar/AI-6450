from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit navigation dataset quality and distribution")
    p.add_argument("--dataset", type=str, default="data_maze_small/navigation_dataset.npz")
    p.add_argument("--output", type=str, default="artifacts/data_audit")
    p.add_argument("--dedupe-round", type=int, default=3, help="Rounding decimals for relative_goal in duplicate check")
    return p


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(np.sum(p), 1e-12)
    q = q / max(np.sum(q), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, None))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    local_patch = data["local_patch"]
    relative_goal = data["relative_goal"]
    action_id = data["action_id"]
    cost_to_goal = data["cost_to_goal"]
    agent_position = data["agent_position"]
    split = data["split"]

    n = int(action_id.shape[0])

    missing = {
        "local_patch_nan": int(np.isnan(local_patch).sum()) if np.issubdtype(local_patch.dtype, np.floating) else 0,
        "relative_goal_nan": int(np.isnan(relative_goal).sum()),
        "cost_to_goal_nan": int(np.isnan(cost_to_goal).sum()),
    }

    unit_checks = {
        "local_patch_binary_ok": bool(np.all((local_patch == 0) | (local_patch == 1))),
        "relative_goal_range_ok": bool(np.all(relative_goal >= -1.05) and np.all(relative_goal <= 1.05)),
        "action_range_ok": bool(np.min(action_id) >= 0 and np.max(action_id) <= 7),
        "agent_position_nonnegative": bool(np.min(agent_position) >= 0),
    }

    # Duplicates: same local patch + rounded relative goal + exact agent position.
    rel_q = np.round(relative_goal, args.dedupe_round)
    flat_patch = local_patch.reshape(n, -1).astype(np.int16)
    dedupe_matrix = np.concatenate([flat_patch, (rel_q * 1000).astype(np.int16), agent_position.astype(np.int16)], axis=1)
    unique_rows = np.unique(dedupe_matrix, axis=0).shape[0]
    duplicate_count = int(n - unique_rows)
    duplicate_ratio = float(duplicate_count / max(n, 1))

    # Outliers in cost/path-length proxy.
    cost_p95 = float(np.percentile(cost_to_goal, 95))
    cost_p99 = float(np.percentile(cost_to_goal, 99))
    outlier_thresh = float(cost_p99)
    outlier_count = int(np.sum(cost_to_goal > outlier_thresh))

    # Class imbalance.
    cls_counts = np.bincount(action_id, minlength=8)
    imbalance_ratio = float(np.max(cls_counts) / max(np.min(cls_counts), 1))

    # Drift check: compare train and test marginals.
    tr = split == 0
    te = split == 2
    tr_counts = np.bincount(action_id[tr], minlength=8)
    te_counts = np.bincount(action_id[te], minlength=8)
    js_action = float(js_divergence(tr_counts, te_counts))
    drift = {
        "train_cost_mean": float(np.mean(cost_to_goal[tr])) if np.any(tr) else float("nan"),
        "test_cost_mean": float(np.mean(cost_to_goal[te])) if np.any(te) else float("nan"),
        "train_cost_std": float(np.std(cost_to_goal[tr])) if np.any(tr) else float("nan"),
        "test_cost_std": float(np.std(cost_to_goal[te])) if np.any(te) else float("nan"),
        "action_js_divergence_train_vs_test": js_action,
    }

    handling_plan = {
        "missing_values": "None expected; verified counts in report",
        "inconsistent_units": "Fixed grid units; binary occupancy and normalized relative goal checked",
        "label_noise": "Very low by construction from deterministic planner",
        "outliers": "Long paths measured via cost percentiles; map size controls tail",
        "duplicate_samples": "Possible; duplicate ratio reported",
        "data_drift": "Train/test drift measured by action JS divergence + cost stats",
        "class_imbalance": "Likely; class count ratio reported (consider weighted loss)",
        "labeling": "Deterministic planner ground truth",
    }

    report: Dict[str, object] = {
        "num_samples": n,
        "missing": missing,
        "unit_checks": unit_checks,
        "duplicates": {
            "duplicate_count": duplicate_count,
            "duplicate_ratio": duplicate_ratio,
        },
        "outliers": {
            "cost_p95": cost_p95,
            "cost_p99": cost_p99,
            "outlier_threshold": outlier_thresh,
            "outlier_count": outlier_count,
        },
        "class_distribution": {str(i): int(c) for i, c in enumerate(cls_counts.tolist())},
        "class_imbalance_ratio_max_over_min": imbalance_ratio,
        "drift": drift,
        "issue_handling_plan_checks": handling_plan,
    }

    with (out_dir / "dataset_audit_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Plots
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(8), cls_counts)
    ax.set_xlabel("action class")
    ax.set_ylabel("count")
    ax.set_title("Action class distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "action_class_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cost_to_goal[tr], bins=40, alpha=0.6, label="train")
    ax.hist(cost_to_goal[te], bins=40, alpha=0.6, label="test")
    ax.set_xlabel("cost_to_goal")
    ax.set_ylabel("count")
    ax.set_title("Cost distribution (train vs test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cost_distribution_train_vs_test.png", dpi=180)
    plt.close(fig)

    print("Dataset audit complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
