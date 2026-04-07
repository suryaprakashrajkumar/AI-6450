# Imitation Learning Path Planner (10x10)

This project trains a neural policy to imitate an 8-direction A* expert on 10x10 grids.

## What is implemented
- Expert data generation with A* over diverse obstacle maps and start/goal pairs.
- Iterative next-action imitation model (CNN + MLP) in PyTorch.
- Lightweight sklearn baseline for sanity comparison.
- Dataset quality gates for:
  - missing values
  - inconsistent formats/units
  - label noise/weak labels
  - outliers/erroneous values
  - duplicate values
  - data drift risk (temporal, marked N/A for static synthetic generation)
  - class imbalance
  - unsupervised labelling concern (marked N/A, labels are from A*)
- End-to-end benchmark against A* solve time on CPU.

## Setup (Windows)

### Conda path (preferred)
1. `conda create -n imitation python=3.10 -y`
1. `conda activate imitation`
1. `pip install -r requirements.txt`

### venv fallback
1. `python -m venv .venv`
1. `.venv\Scripts\activate`
1. `pip install -r requirements.txt`

## Run pipeline
1. Generate dataset:
   - `python scripts/generate_data.py`
1. Run QA checks:
   - `python scripts/quality_check.py`
1. Train PyTorch model:
   - `python scripts/train_model.py`
1. Train sklearn baseline:
   - `python scripts/run_baseline.py`
1. Evaluate policy vs A*:
   - `python scripts/evaluate_model.py`

## Training runbook (recommended)

### 1. Smoke training/eval
1. Generate smoke dataset:
   - `python -m scripts.generate_data --config config_smoke.json`
1. Quality check:
   - `python -m scripts.quality_check --config config_smoke.json`
1. Train base model:
   - `python -m scripts.train_model --config config_smoke.json --output models/final/imitation_policy_smoke.pth --metrics_output logs/smoke/train_metrics.json --history_output logs/smoke/train_history.csv --history_plot_output logs/smoke/training_curves.png`
1. Evaluate base model:
   - `python -m scripts.evaluate_model --config config_smoke.json --test data/processed/test_rollout.npz --test_steps data/processed/test_samples.npz --model models/final/imitation_policy_smoke.pth --metrics_output logs/smoke/rollout_kpi.json`

### 2. Scale-up base model (required before DAgger/ER-CL)
1. Generate scale-up dataset:
   - `python -m scripts.generate_data --config config_scaleup.json`
1. Quality check:
   - `python -m scripts.quality_check --config config_scaleup.json`
1. Train scale-up base checkpoint:
   - `python -m scripts.train_model --config config_scaleup.json --output models/final/imitation_policy_scaleup.pth --metrics_output logs/scaleup_run/train_metrics.json --history_output logs/scaleup_run/train_history.csv --history_plot_output logs/scaleup_run/training_curves.png`

## DAgger adaptation

Run DAgger from the scale-up base model:

- `python -m scripts.run_dagger --config config_scaleup.json --base_model models/final/imitation_policy_scaleup.pth --train_rollout data/processed_scaleup/train_rollout.npz --train_steps data/processed_scaleup/train_samples.npz --val_steps data/processed_scaleup/val_samples.npz --test_rollout data/processed_scaleup/test_rollout.npz --test_steps data/processed_scaleup/test_samples.npz --out_dir logs/dagger_scaleup --output_model models/final/imitation_policy_scaleup_dagger.pth`

Main outputs:
- `models/final/imitation_policy_scaleup_dagger.pth`
- `logs/dagger_scaleup/baseline_eval_kpi.json`
- `logs/dagger_scaleup/dagger_eval_kpi.json`
- `logs/dagger_scaleup/baseline_eval_kpi_no_fallback.json`
- `logs/dagger_scaleup/dagger_eval_kpi_no_fallback.json`
- `logs/dagger_scaleup/dagger_summary.json`

## ER-CL adaptation (Experience Replay Continual Learning)

Run replay-based continual learning from the same base model:

- `python -m scripts.run_experience_replay_cl --config config_scaleup.json --base_model models/final/imitation_policy_scaleup.pth --train_rollout data/processed_scaleup/train_rollout.npz --train_steps data/processed_scaleup/train_samples.npz --val_steps data/processed_scaleup/val_samples.npz --test_rollout data/processed_scaleup/test_rollout.npz --test_steps data/processed_scaleup/test_samples.npz --out_dir logs/replay_cl_scaleup --output_model models/final/imitation_policy_scaleup_replay_cl.pth`

Main outputs:
- `models/final/imitation_policy_scaleup_replay_cl.pth`
- `logs/replay_cl_scaleup/baseline_eval_kpi.json`
- `logs/replay_cl_scaleup/replay_eval_kpi.json`
- `logs/replay_cl_scaleup/baseline_eval_kpi_no_fallback.json`
- `logs/replay_cl_scaleup/replay_eval_kpi_no_fallback.json`
- `logs/replay_cl_scaleup/replay_summary.json`

## Quick comparison of base vs DAgger vs ER-CL
- `python -m scripts.evaluate_model --config config_scaleup.json --model models/final/imitation_policy_scaleup.pth --test data/processed_scaleup/test_rollout.npz --test_steps data/processed_scaleup/test_samples.npz --metrics_output logs/adaptation/base_eval.json`
- `python -m scripts.evaluate_model --config config_scaleup.json --model models/final/imitation_policy_scaleup_dagger.pth --test data/processed_scaleup/test_rollout.npz --test_steps data/processed_scaleup/test_samples.npz --metrics_output logs/adaptation/dagger_eval.json`
- `python -m scripts.evaluate_model --config config_scaleup.json --model models/final/imitation_policy_scaleup_replay_cl.pth --test data/processed_scaleup/test_rollout.npz --test_steps data/processed_scaleup/test_samples.npz --metrics_output logs/adaptation/replay_cl_eval.json`

## KPI suite (implemented)
The evaluation now reports these KPIs:
- `path_validity`: fraction of test rollout tasks reaching goal without invalid moves.
- `exact_match`: fraction of tasks where predicted path exactly matches A* path.
- `step_accuracy`: next-action classification accuracy on test step dataset.
- `optimality_gap_mean`: mean `(pred_len - astar_len) / astar_len` over successful rollouts.
- `optimality_gap_p95`: 95th percentile of optimality gap.
- `success_under_1_5x`: successful rollouts with path length <= 1.5x A*.
- `path_length_ratio_mean`: mean `pred_len / astar_len` for successful rollouts.
- `astar_ms_per_query`: average A* solve time.
- `policy_ms_per_query`: average end-to-end policy rollout solve time.
- `speedup_astar_over_policy`: A* time divided by policy time.
- `speedup_policy_over_astar`: policy time divided by A* time.

## Large run and ablation (implemented)

### 1. Large-data generation and QA
- `python -m scripts.generate_data --config config_large.json`
- `python -m scripts.quality_check --config config_large.json`

### 2. Ablation across model families
- `python -m scripts.ablation_study --config config_large.json --models mlp,cnn_small,cnn_large --out_dir logs/ablation --include_sklearn`

Outputs:
- `logs/ablation/ablation_results.csv`
- Per-model train/eval metrics JSON files and model checkpoints

### 3. Charts and rollout-time comparisons
- `python -m scripts.plot_ablation --input logs/ablation/ablation_results.csv --out_dir logs/ablation/charts`

Generated charts:
- `kpi_comparison.png`
- `rollout_time_comparison.png`
- `optimality_gap_comparison.png`

### 4. Single-command full experiment run
- `python -m scripts.run_full_experiments --config config_large.json --models mlp,cnn_small,cnn_large --out_dir logs/ablation`

## Smoke validation already executed
- Smoke ablation run completed at `logs/ablation_smoke`.
- KPI CSV and charts were generated successfully.
- Unit tests pass (`5 passed`).

## Improved high-success run (executed)

Commands:
- `python -m scripts.generate_data --config config_improved.json`
- `python -m scripts.quality_check --config config_improved.json`
- `python -m scripts.train_model --config config_improved.json --output models/final/imitation_policy_improved.pth --metrics_output logs/improved_run/train_metrics.json --history_output logs/improved_run/train_history.csv --history_plot_output logs/improved_run/training_curves.png`
- `python -m scripts.evaluate_model --config config_improved.json --test data/processed/test_rollout.npz --test_steps data/processed/test_samples.npz --model models/final/imitation_policy_improved.pth --metrics_output logs/improved_run/rollout_kpi.json`

Saved artifacts:
- Training log: `logs/improved_run/train_output.txt`
- Training metrics: `logs/improved_run/train_metrics.json`
- Training history CSV: `logs/improved_run/train_history.csv`
- Training graphs: `logs/improved_run/training_curves.png`
- Rollout KPI JSON: `logs/improved_run/rollout_kpi.json`
- Rollout eval console output: `logs/improved_run/rollout_eval_output.txt`

## Core paths
- `src/environment.py`: A* expert and grid dynamics.
- `src/dataset.py`: dataset loader and quality checks.
- `src/model.py`: imitation policy network.
- `src/train.py`: training loop with early stopping.
- `src/eval.py`: rollout and benchmark metrics.

## Notes
- Speed metric is end-to-end rollout solve time on CPU.
- If strict speedup over A* is not reached for all cases, a practical hybrid strategy is to use policy-first with A* fallback on confidence/step-cap triggers.
