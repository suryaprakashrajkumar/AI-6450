# Project Guidelines

## Current Functionality And Objective
- Current functionality: synthetic grid-world data generation with A* expert labels, strict dataset QA, supervised next-action imitation training, rollout-based evaluation, sklearn baseline, and experiment scripts for ablation/robustness/monitoring.
- Theory: iterative next-action imitation (not whole-path prediction) on 10x10 maps with 8-direction actions, using map-level train/val/test split to reduce leakage and improve generalization to unseen maps.
- Objective: produce a deployment-credible policy that keeps path validity high, optimality gap low vs A*, and end-to-end CPU solve time competitive, with fallback usage explicitly reported when enabled.

## Build And Test
- Create environment and install dependencies: `python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`.
- Preferred quick validation flow:
  - `python scripts/generate_data.py --config config_smoke.json`
  - `python scripts/quality_check.py --config config_smoke.json`
  - `python scripts/train_model.py --config config_smoke.json`
  - `python scripts/evaluate_model.py --config config_smoke.json`
- Run tests with `pytest tests/ -v` before finalizing non-trivial changes.
- For full experiments, use `config_improved.json` or `config_large.json`.

## Architecture
- `src/environment.py`: `GridWorld`, A* expert planner, and 8-direction action mapping.
- `src/dataset.py`: NPZ loading, coordinate normalization to [0, 1], and dataset quality checks.
- `src/model.py`: policy model variants (`PathPolicyNet`, `PathPolicyNetLarge`, `PathPolicyMLP`) and model factory.
- `src/train.py`: training orchestration with class-weighted loss, scheduler, and early stopping.
- `src/eval.py`: rollout metrics, speed metrics, and optional confidence-based A* fallback.
- `scripts/*.py`: thin orchestration wrappers around `src` logic for data generation, QA, train, baseline, eval, and studies.

## Conventions
- Keep the 10x10 grid contract unless a task explicitly expands grid size across the stack.
- Keep action space as 8-direction moves; preserve action ID and delta mapping semantics.
- Split data by map identity, not by individual step samples.
- Preserve position normalization behavior (coordinates normalized to [0, 1] in dataset loading).
- Treat dataset QA as fail-fast for training data integrity.

## Pitfalls
- Avoid train/test leakage by reusing map-level split logic.
- Do not bypass QA checks in normal training paths.
- Be explicit when enabling fallback in evaluation, since fallback can mask policy failures.

## References
- See `README.md` for full workflow, KPI definitions (`path_validity`, `optimality_gap_*`, speed metrics), and experiment commands.
- See `config_smoke.json`, `config_improved.json`, and `config_large.json` for canonical run profiles.