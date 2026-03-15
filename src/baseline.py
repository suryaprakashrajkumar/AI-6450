from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def _flatten_features(npz_path: str):
    payload = np.load(npz_path)
    grids = payload["grids"].reshape(len(payload["grids"]), -1).astype(np.float32)
    positions = payload["positions"].astype(np.float32) / 9.0
    goals = payload["goals"].astype(np.float32) / 9.0
    x = np.concatenate([grids, positions, goals], axis=1)
    y = payload["actions"].astype(np.int64)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/processed/train_samples.npz")
    parser.add_argument("--val", type=str, default="data/processed/val_samples.npz")
    parser.add_argument("--output", type=str, default="models/final/sklearn_baseline.joblib")
    args = parser.parse_args()

    x_train, y_train = _flatten_features(args.train)
    x_val, y_val = _flatten_features(args.val)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=100, random_state=42)
    clf.fit(x_train, y_train)

    pred = clf.predict(x_val)
    acc = accuracy_score(y_val, pred)
    print(f"baseline_val_acc={acc:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out)
    print(f"Saved sklearn baseline to {out}")


if __name__ == "__main__":
    main()
