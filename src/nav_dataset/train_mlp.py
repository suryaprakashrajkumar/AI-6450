from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train simple MLP models for action and cost prediction")
    p.add_argument("--dataset", type=str, default="data_maze_small/navigation_dataset.npz")
    p.add_argument("--output", type=str, default="mlp_runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=30000)
    p.add_argument("--max-eval-samples", type=int, default=12000)
    p.add_argument("--show-inference", type=int, default=10)
    return p


def build_features(data: np.lib.npyio.NpzFile) -> np.ndarray:
    patch = data["local_patch"].astype(np.float32).reshape(data["local_patch"].shape[0], -1)
    rel_goal = data["relative_goal"].astype(np.float32)
    agent_pos = data["agent_position"].astype(np.float32)

    max_row = np.maximum(np.max(agent_pos[:, 0]), 1.0)
    max_col = np.maximum(np.max(agent_pos[:, 1]), 1.0)
    agent_pos[:, 0] /= max_row
    agent_pos[:, 1] /= max_col

    x = np.concatenate([patch, rel_goal, agent_pos], axis=1)
    return x.astype(np.float32)


def sample_indices(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if idx.size <= cap:
        return idx
    return rng.choice(idx, size=cap, replace=False)


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    x_train = x[train_idx]
    x_val = x[val_idx]
    x_test = x[test_idx]

    y_act_train = y_action[train_idx]
    y_act_val = y_action[val_idx]
    y_act_test = y_action[test_idx]

    y_cost_train = y_cost[train_idx]
    y_cost_val = y_cost[val_idx]
    y_cost_test = y_cost[test_idx]

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=40,
                    early_stopping=True,
                    n_iter_no_change=5,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    reg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=50,
                    early_stopping=True,
                    n_iter_no_change=6,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    clf.fit(x_train, y_act_train)
    reg.fit(x_train, y_cost_train)

    pred_val_act = clf.predict(x_val)
    pred_test_act = clf.predict(x_test)

    pred_val_cost = reg.predict(x_val)
    pred_test_cost = reg.predict(x_test)

    metrics = {
        "classification": {
            "val_accuracy": float(accuracy_score(y_act_val, pred_val_act)),
            "test_accuracy": float(accuracy_score(y_act_test, pred_test_act)),
        },
        "regression": {
            "val_mae": float(mean_absolute_error(y_cost_val, pred_val_cost)),
            "test_mae": float(mean_absolute_error(y_cost_test, pred_test_cost)),
            "val_rmse": float(np.sqrt(mean_squared_error(y_cost_val, pred_val_cost))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_cost_test, pred_test_cost))),
        },
        "data": {
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
        },
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (out_dir / "action_mlp.pkl").open("wb") as f:
        pickle.dump(clf, f)

    with (out_dir / "cost_mlp.pkl").open("wb") as f:
        pickle.dump(reg, f)

    print("Training complete")
    print(json.dumps(metrics, indent=2))

    # Show inference examples from held-out test set.
    n_show = min(args.show_inference, x_test.shape[0])
    show_idx = rng.choice(np.arange(x_test.shape[0]), size=n_show, replace=False)
    probs = clf.predict_proba(x_test[show_idx])
    pred_actions = clf.predict(x_test[show_idx])
    pred_costs = reg.predict(x_test[show_idx])

    print("\nSample inferences (test split):")
    for i, local_i in enumerate(show_idx):
        global_idx = int(test_idx[local_i])
        true_a = int(y_act_test[local_i])
        pred_a = int(pred_actions[i])
        conf = float(np.max(probs[i]))
        true_c = float(y_cost_test[local_i])
        pred_c = float(pred_costs[i])
        print(
            f"idx={global_idx} | action true/pred={true_a}/{pred_a} (conf={conf:.3f}) "
            f"| cost true/pred={true_c:.2f}/{pred_c:.2f}"
        )


if __name__ == "__main__":
    main()
