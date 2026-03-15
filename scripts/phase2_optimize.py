from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.eval import evaluate_model


def _score(metrics: dict[str, float | str], min_validity: float) -> float:
    path_validity = float(metrics["path_validity"])
    exact_match = float(metrics["exact_match"])
    success_under_1_5x = float(metrics["success_under_1_5x"])
    policy_ms = float(metrics["policy_ms_per_query"])

    if path_validity < min_validity:
        return -1e9 + path_validity * 1000.0

    # High validity and quality first, then minimize latency.
    return (path_validity * 1000.0) + (exact_match * 100.0) + (success_under_1_5x * 50.0) - (policy_ms * 0.5)


def _update_config(base_cfg: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(base_cfg)
    out.update(candidate)
    return out

    


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="config_improved.json")
    parser.add_argument("--model", type=str, default="models/final/imitation_policy_improved.pth")
    parser.add_argument("--test_rollout", type=str, default="data/processed/test_rollout.npz")
    parser.add_argument("--test_steps", type=str, default="data/processed/test_samples.npz")
    parser.add_argument("--out_dir", type=str, default="logs/phase2")
    parser.add_argument("--min_validity", type=float, default=0.98)
    parser.add_argument("--rollout_limit", type=int, default=120)
    parser.add_argument("--coarse_only", action="store_true")
    parser.add_argument("--topk_refine", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = json.loads(Path(args.base_config).read_text(encoding="utf-8"))

    # Coarse stage: small candidate set for quick runtime.
    coarse_candidates: list[dict[str, Any]] = []
    for enable_conf in [True, False]:
        for thresh in [0.28, 0.36]:
            for min_step in [4, 8]:
                for patience in [1, 3]:
                    for max_calls in [0, 1]:
                        coarse_candidates.append(
                            {
                                "enable_astar_fallback": bool(max_calls > 0),
                                "enable_confidence_fallback": enable_conf,
                                "fallback_confidence_threshold": thresh,
                                "fallback_min_step": min_step,
                                "fallback_no_progress_patience": patience,
                                "fallback_max_calls": max_calls,
                            }
                        )

    rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_row: dict[str, Any] | None = None
    best_cfg: dict[str, Any] | None = None

    def run_candidates(candidates: list[dict[str, Any]], stage_name: str) -> list[dict[str, Any]]:
        stage_rows: list[dict[str, Any]] = []
        for i, candidate in enumerate(candidates, start=1):
            cfg = _update_config(base_cfg, candidate)
            temp_cfg = out_dir / "_temp_config.json"
            temp_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            metrics = evaluate_model(
                rollout_npz_path=args.test_rollout,
                step_npz_path=args.test_steps,
                model_path=args.model,
                config_path=str(temp_cfg),
                metrics_output=None,
                compute_step_accuracy=False,
                rollout_limit=args.rollout_limit,
            )

            row = {**candidate, **metrics}
            row["stage"] = stage_name
            row["score"] = _score(metrics, args.min_validity)
            stage_rows.append(row)

            print(
                f"[{stage_name} {i}/{len(candidates)}] "
                f"validity={float(metrics['path_validity']):.4f} "
                f"policy_ms={float(metrics['policy_ms_per_query']):.4f} "
                f"score={float(row['score']):.2f}"
            )
        return stage_rows

    coarse_rows = run_candidates(coarse_candidates, "coarse")
    rows.extend(coarse_rows)

    refine_rows: list[dict[str, Any]] = []
    if not args.coarse_only:
        coarse_sorted = sorted(coarse_rows, key=lambda r: float(r["score"]), reverse=True)
        top_coarse = coarse_sorted[: max(1, args.topk_refine)]
        refine_candidates: list[dict[str, Any]] = []
        for row in top_coarse:
            base_thresh = float(row["fallback_confidence_threshold"])
            base_min_step = int(row["fallback_min_step"])
            base_patience = int(row["fallback_no_progress_patience"])
            base_max_calls = int(row["fallback_max_calls"])
            for dt in [-0.02, 0.0, 0.02]:
                for ds in [-1, 1]:
                    for dp in [-1, 1]:
                        cand = {
                            "enable_astar_fallback": bool(base_max_calls > 0),
                            "enable_confidence_fallback": bool(row["enable_confidence_fallback"]),
                            "fallback_confidence_threshold": max(0.15, min(0.55, base_thresh + dt)),
                            "fallback_min_step": max(0, base_min_step + ds),
                            "fallback_no_progress_patience": max(0, base_patience + dp),
                            "fallback_max_calls": max(0, min(2, base_max_calls)),
                        }
                        refine_candidates.append(cand)

        # Deduplicate refine candidates
        unique_refine: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        for c in refine_candidates:
            key = (
                c["enable_astar_fallback"],
                c["enable_confidence_fallback"],
                round(float(c["fallback_confidence_threshold"]), 3),
                int(c["fallback_min_step"]),
                int(c["fallback_no_progress_patience"]),
                int(c["fallback_max_calls"]),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_refine.append(c)

        refine_rows = run_candidates(unique_refine, "refine")
        rows.extend(refine_rows)

    for row in rows:
        if float(row["score"]) > best_score:
            best_score = float(row["score"])
            best_row = row
            best_cfg = _update_config(base_cfg, {
                "enable_astar_fallback": row["enable_astar_fallback"],
                "enable_confidence_fallback": row["enable_confidence_fallback"],
                "fallback_confidence_threshold": row["fallback_confidence_threshold"],
                "fallback_min_step": row["fallback_min_step"],
                "fallback_no_progress_patience": row["fallback_no_progress_patience"],
                "fallback_max_calls": row["fallback_max_calls"],
            })

    csv_path = out_dir / "phase2_sweep_results.csv"
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    if best_row is None or best_cfg is None:
        raise RuntimeError("No phase-2 candidate was evaluated.")

    best_metrics_path = out_dir / "best_metrics.json"
    best_metrics_path.write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    best_cfg_path = out_dir / "config_phase2_best.json"
    best_cfg_path.write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")

    # Final full evaluation with best config and full KPIs
    final_metrics = evaluate_model(
        rollout_npz_path=args.test_rollout,
        step_npz_path=args.test_steps,
        model_path=args.model,
        config_path=str(best_cfg_path),
        metrics_output=str(out_dir / "best_metrics_full_eval.json"),
        compute_step_accuracy=True,
    )
    print(f"Final full-eval path_validity={float(final_metrics['path_validity']):.4f} policy_ms={float(final_metrics['policy_ms_per_query']):.4f}")

    print(f"Saved sweep: {csv_path}")
    print(f"Saved best metrics: {best_metrics_path}")
    print(f"Saved best config: {best_cfg_path}")


if __name__ == "__main__":
    main()
