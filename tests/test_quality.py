from pathlib import Path

import numpy as np

from src.dataset import run_quality_checks


def test_quality_checks_pass_for_clean_data(tmp_path: Path):
    n = 8
    grids = np.zeros((n, 10, 10), dtype=np.int8)
    positions = np.array([[i, i] for i in range(n)], dtype=np.int16)
    goals = np.array([[i + 1, i + 1] for i in range(n)], dtype=np.int16)
    actions = np.full((n,), 3, dtype=np.int64)

    path = tmp_path / "clean.npz"
    np.savez_compressed(path, grids=grids, positions=positions, goals=goals, actions=actions)

    report = run_quality_checks(str(path), imbalance_threshold=0.0)
    assert report.passed
