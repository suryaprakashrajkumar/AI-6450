from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    lo = min(float(ref.min()), float(cur.min()))
    hi = max(float(ref.max()), float(cur.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_p = ref_hist / max(1, ref_hist.sum())
    cur_p = cur_hist / max(1, cur_hist.sum())

    eps = 1e-6
    ref_p = np.clip(ref_p, eps, 1.0)
    cur_p = np.clip(cur_p, eps, 1.0)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _js_div(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    ref_p = ref_counts / max(1, ref_counts.sum())
    cur_p = cur_counts / max(1, cur_counts.sum())
    m = 0.5 * (ref_p + cur_p)
    eps = 1e-9
    ref_p = np.clip(ref_p, eps, 1.0)
    cur_p = np.clip(cur_p, eps, 1.0)
    m = np.clip(m, eps, 1.0)
    kl_ref_m = np.sum(ref_p * np.log(ref_p / m))
    kl_cur_m = np.sum(cur_p * np.log(cur_p / m))
    return float(0.5 * (kl_ref_m + kl_cur_m))


def _window_iter(n: int, window: int, stride: int):
    s = 0
    while s + window <= n:
        yield s, s + window
        s += stride


def _build_stream(base: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = base.copy()
    n = len(out)
    a = int(0.4 * n)
    b = int(0.75 * n)

    # Window 2: moderate shift
    for i in range(a, b):
        p = 0.12
        mask = rng.random((10, 10)) < p
        out[i][mask] = 1

    # Window 3: severe shift
    for i in range(b, n):
        p = 0.22
        mask = rng.random((10, 10)) < p
        out[i][mask] = 1

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_steps", type=str, default="data/processed/train_samples.npz")
    parser.add_argument("--stream_steps", type=str, default="data/processed/test_samples.npz")
    parser.add_argument("--out_dir", type=str, default="logs/monitoring")
    parser.add_argument("--window", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--psi_alert", type=float, default=0.2)
    parser.add_argument("--js_alert", type=float, default=0.08)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = np.load(args.reference_steps)
    stream = np.load(args.stream_steps)

    ref_grids = ref["grids"].astype(np.float32)
    ref_actions = ref["actions"].astype(np.int64)

    stream_grids = stream["grids"].astype(np.float32)
    stream_actions = stream["actions"].astype(np.int64)

    stream_grids = _build_stream(stream_grids, seed=args.seed)

    ref_density = ref_grids.mean(axis=(1, 2))
    ref_action_counts = np.bincount(ref_actions, minlength=8)

    rows: list[dict[str, float | str]] = []
    for wid, (s, e) in enumerate(_window_iter(len(stream_grids), args.window, args.stride), start=1):
        w_grids = stream_grids[s:e]
        w_actions = stream_actions[s:e]

        w_density = w_grids.mean(axis=(1, 2))
        w_action_counts = np.bincount(w_actions, minlength=8)

        psi = _psi(ref_density, w_density, bins=10)
        js = _js_div(ref_action_counts.astype(np.float64), w_action_counts.astype(np.float64))
        density_mean = float(np.mean(w_density))

        alert = []
        if psi > args.psi_alert:
            alert.append("psi")
        if js > args.js_alert:
            alert.append("js")

        rows.append(
            {
                "window_id": float(wid),
                "start": float(s),
                "end": float(e),
                "density_mean": density_mean,
                "psi_density": float(psi),
                "js_action": float(js),
                "alert": "|".join(alert) if alert else "none",
            }
        )

    csv_path = out_dir / "monitoring_windows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["window_id", "start", "end", "density_mean", "psi_density", "js_action", "alert"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Dashboard plot
    x = [int(r["window_id"]) for r in rows]
    psi_vals = [float(r["psi_density"]) for r in rows]
    js_vals = [float(r["js_action"]) for r in rows]
    dens = [float(r["density_mean"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(x, dens, marker="o", color="#1D4ED8")
    axes[0].set_title("Density Mean by Window")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, psi_vals, marker="o", color="#B45309")
    axes[1].axhline(args.psi_alert, linestyle="--", color="#DC2626", label="PSI alert threshold")
    axes[1].set_title("PSI (Density) by Window")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, js_vals, marker="o", color="#047857")
    axes[2].axhline(args.js_alert, linestyle="--", color="#DC2626", label="JS alert threshold")
    axes[2].set_title("JS (Action Distribution) by Window")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("window id")

    fig.tight_layout()
    fig.savefig(out_dir / "monitoring_dashboard.png", dpi=170)
    plt.close(fig)

    alert_windows = [r for r in rows if r["alert"] != "none"]
    summary = {
        "n_windows": len(rows),
        "n_alert_windows": len(alert_windows),
        "psi_alert_threshold": args.psi_alert,
        "js_alert_threshold": args.js_alert,
    }
    (out_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
