from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.environment import AStarPlanner, Point, is_valid_transition


@dataclass
class QAReport:
    passed: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, float]


class ImitationDataset(Dataset):
    def __init__(self, npz_path: str):
        payload = np.load(npz_path)
        self.grids = payload["grids"].astype(np.float32)
        self.positions = payload["positions"].astype(np.float32)
        self.goals = payload["goals"].astype(np.float32)
        self.actions = payload["actions"].astype(np.int64)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        grid = torch.from_numpy(self.grids[idx]).unsqueeze(0)
        pos = torch.from_numpy(self.positions[idx]) / 9.0
        goal = torch.from_numpy(self.goals[idx]) / 9.0
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        return grid, pos, goal, action


def _check_missing_values(grids: np.ndarray, positions: np.ndarray, goals: np.ndarray, actions: np.ndarray) -> list[str]:
    errors: list[str] = []
    if np.isnan(grids).any() or np.isnan(positions).any() or np.isnan(goals).any():
        errors.append("Missing values: NaN detected in features.")
    if np.isnan(actions.astype(np.float64)).any():
        errors.append("Missing values: NaN detected in labels.")
    return errors


def _check_formats(grids: np.ndarray, positions: np.ndarray, goals: np.ndarray, actions: np.ndarray) -> list[str]:
    errors: list[str] = []
    if grids.ndim != 3 or grids.shape[1:] != (10, 10):
        errors.append(f"Inconsistent format: grids must be [N,10,10], got {grids.shape}.")
    if positions.ndim != 2 or positions.shape[1] != 2:
        errors.append(f"Inconsistent format: positions must be [N,2], got {positions.shape}.")
    if goals.ndim != 2 or goals.shape[1] != 2:
        errors.append(f"Inconsistent format: goals must be [N,2], got {goals.shape}.")
    if actions.ndim != 1:
        errors.append(f"Inconsistent format: actions must be [N], got {actions.shape}.")

    if not (np.logical_and(positions >= 0, positions <= 9).all() and np.logical_and(goals >= 0, goals <= 9).all()):
        errors.append("Inconsistent units: coordinates out of [0,9] bounds.")
    if not np.logical_and(actions >= 0, actions < 8).all():
        errors.append("Inconsistent units: action ids out of [0,7].")
    return errors


def _check_duplicates(grids: np.ndarray, positions: np.ndarray, goals: np.ndarray, actions: np.ndarray) -> tuple[int, int]:
    seen: set[str] = set()
    dupes = 0
    for i in range(len(actions)):
        h = sha1()
        h.update(grids[i].astype(np.int8).tobytes())
        h.update(positions[i].astype(np.int16).tobytes())
        h.update(goals[i].astype(np.int16).tobytes())
        h.update(np.array([actions[i]], dtype=np.int8).tobytes())
        key = h.hexdigest()
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
    return dupes, len(actions)


def _check_class_imbalance(actions: np.ndarray, threshold: float) -> tuple[dict[int, int], str | None]:
    counts = np.bincount(actions, minlength=8)
    min_count = int(counts.min())
    max_count = int(counts.max())
    ratio = (min_count / max_count) if max_count > 0 else 0.0
    warning = None
    if ratio < threshold:
        warning = f"Class imbalance: min/max frequency ratio {ratio:.3f} < {threshold:.3f}."
    return {i: int(v) for i, v in enumerate(counts)}, warning


def _check_label_consistency(
    grids: np.ndarray,
    positions: np.ndarray,
    goals: np.ndarray,
    actions: np.ndarray,
    sample_size: int = 300,
) -> list[str]:
    planner = AStarPlanner()
    errors: list[str] = []
    n = len(actions)
    if n == 0:
        return ["Label quality: empty dataset."]

    idxs = np.linspace(0, n - 1, num=min(sample_size, n), dtype=int)
    for idx in idxs:
        grid = grids[idx]
        start = Point(int(positions[idx][0]), int(positions[idx][1]))
        goal = Point(int(goals[idx][0]), int(goals[idx][1]))

        if not is_valid_transition(grid, start, int(actions[idx])):
            errors.append(f"Erroneous value: invalid transition at sample {idx}.")
            continue

        path = planner.solve(grid, start, goal)
        if path is None or len(path) < 2:
            errors.append(f"Weak label: no expert path available at sample {idx}.")
            continue

        delta = (path[1].row - path[0].row, path[1].col - path[0].col)
        expected = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)].index(delta)
        if expected != int(actions[idx]):
            errors.append(
                f"Label noise: sample {idx} action mismatch expert ({actions[idx]} vs {expected})."
            )
            if len(errors) > 50:
                break
    return errors


def run_quality_checks(npz_path: str, imbalance_threshold: float = 0.15) -> QAReport:
    payload = np.load(npz_path)
    grids = payload["grids"]
    positions = payload["positions"]
    goals = payload["goals"]
    actions = payload["actions"]

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_check_missing_values(grids, positions, goals, actions))
    errors.extend(_check_formats(grids, positions, goals, actions))

    dupes, total = _check_duplicates(grids, positions, goals, actions)
    if dupes > 0:
        warnings.append(f"Duplicate values: {dupes}/{total} duplicate samples found.")

    counts, imbalance_warning = _check_class_imbalance(actions, imbalance_threshold)
    if imbalance_warning is not None:
        warnings.append(imbalance_warning)

    errors.extend(_check_label_consistency(grids, positions, goals, actions))

    # Synthetic static generation has no temporal dimension, so drift risk is N/A by design.
    warnings.append("Data drift risk: temporal drift not applicable for static synthetic generation.")
    warnings.append("Labelling mode: supervised expert labels from A* (unsupervised concern not applicable).")

    stats = {
        "n_samples": float(len(actions)),
        "duplicate_ratio": (dupes / total) if total > 0 else 0.0,
        "action_min_count": float(min(counts.values()) if counts else 0),
        "action_max_count": float(max(counts.values()) if counts else 0),
    }
    passed = len(errors) == 0
    return QAReport(passed=passed, errors=errors, warnings=warnings, stats=stats)


def write_qa_report(report: QAReport, output_path: str) -> None:
    lines = [
        f"passed: {report.passed}",
        "",
        "stats:",
    ]
    for k, v in report.stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("warnings:")
    if report.warnings:
        lines.extend([f"  - {w}" for w in report.warnings])
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("errors:")
    if report.errors:
        lines.extend([f"  - {e}" for e in report.errors])
    else:
        lines.append("  - none")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
