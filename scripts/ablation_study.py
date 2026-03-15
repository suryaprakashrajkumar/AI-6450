from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from src.baseline import _flatten_features
from src.eval import evaluate_model
from src.train import train_model


def _hidden_dim_for(model_type: str) -> int:
    if model_type == "cnn_large":
        return 512
    if model_type == "cnn_small":
        return 256
    return 256


def _run_sklearn_baseline(train_npz: str, val_npz: str, test_npz: str, out_model: str) -> dict[str, float | str]:
    from sklearn.neural_network import MLPClassifier

    x_train, y_train = _flatten_features(train_npz)
    x_val, y_val = _flatten_features(val_npz)
    x_test, y_test = _flatten_features(test_npz)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=120, random_state=42)
    clf.fit(x_train, y_train)

    val_acc = accuracy_score(y_val, clf.predict(x_val))
    test_step_acc = accuracy_score(y_test, clf.predict(x_test))

    out_model_path = Path(out_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_model_path)

    return {
        "model_type": "sklearn_mlp",
        "hidden_dim": 0.0,
        "best_val_acc": float(val_acc),
        "best_val_loss": float("nan"),
        "epochs_completed": float("nan"),
        "train_samples": float(len(y_train)),
        "val_samples": float(len(y_val)),
        "samples": float(len(y_test)),
        "path_validity": float("nan"),
        "exact_match": float("nan"),
        "step_accuracy": float(test_step_acc),
        "optimality_gap_mean": float("nan"),
        "optimality_gap_p95": float("nan"),
        "success_under_1_5x": float("nan"),
        "path_length_ratio_mean": float("nan"),
        "astar_ms_per_query": float("nan"),
        "policy_ms_per_query": float("nan"),
        "speedup_astar_over_policy": float("nan"),
        "speedup_policy_over_astar": float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_large.json")
    parser.add_argument("--train", type=str, default="data/processed/train_samples.npz")
    parser.add_argument("--val", type=str, default="data/processed/val_samples.npz")
    parser.add_argument("--test_steps", type=str, default="data/processed/test_samples.npz")
    parser.add_argument("--test_rollout", type=str, default="data/processed/test_rollout.npz")
    parser.add_argument("--models", type=str, default="mlp,cnn_small,cnn_large")
    parser.add_argument("--out_dir", type=str, default="logs/ablation")
    parser.add_argument("--include_sklearn", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []

    for model_type in [m.strip() for m in args.models.split(",") if m.strip()]:
        model_path = out_dir / f"{model_type}.pth"
        train_metrics_path = out_dir / f"{model_type}_train_metrics.json"
        eval_metrics_path = out_dir / f"{model_type}_eval_metrics.json"

        train_metrics = train_model(
            config_path=args.config,
            train_npz=args.train,
            val_npz=args.val,
            output_path=str(model_path),
            model_type=model_type,
            hidden_dim=_hidden_dim_for(model_type),
            metrics_output=str(train_metrics_path),
        )

        eval_metrics = evaluate_model(
            rollout_npz_path=args.test_rollout,
            model_path=str(model_path),
            config_path=args.config,
            step_npz_path=args.test_steps,
            metrics_output=str(eval_metrics_path),
        )

        rows.append({**train_metrics, **eval_metrics})

    if args.include_sklearn:
        baseline_row = _run_sklearn_baseline(
            train_npz=args.train,
            val_npz=args.val,
            test_npz=args.test_steps,
            out_model=str(out_dir / "sklearn_baseline.joblib"),
        )
        rows.append(baseline_row)

    output_csv = out_dir / "ablation_results.csv"
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote ablation results to {output_csv}")


if __name__ == "__main__":
    main()
