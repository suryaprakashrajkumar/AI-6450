# AI-6450

Dataset collection toolkit for studying decision-making methods for autonomous navigation in a 2-D grid world.

This repo is organized as a clean ML project and currently retains the **small maze dataset** as the primary dataset version.

This repository now includes a reproducible data generator for:

- **Heuristic search (A\*)** as expert policy
- **Supervised learning** labels for:
	- next-action classification (8-connected motion)
	- cost-to-goal regression
- **Imitation learning** trajectories from expert demonstrations
- **Maze-based grid path learning** with complete path retention

## Problem setup

At each decision step, a sample includes:

- `local_patch`: fixed-size occupancy map centered at the agent (`1=obstacle`, `0=free`)
- `relative_goal`: goal location relative to agent, normalized to `[-1, 1]`
- labels:
	- `action_id`: one of 8 neighboring directions
	- `cost_to_goal`: shortest-path distance to goal under 8-connected moves

## Dataset collection strategy implemented

The collector follows this strategy:

1. Generate valid occupancy grids with either:
	- `random` obstacle maps, or
	- `maze` maps (complex corridor-like structure)
2. Sample start/goal pairs on free cells.
3. Run A\* to obtain an optimal expert path (if no path exists, reject episode).
4. Convert each state along the path into a supervised sample:
	 - extract local occupancy patch around agent
	 - compute relative goal vector
	 - compute next action from consecutive states
	 - compute exact remaining path cost (suffix length)
5. Save all samples with episode metadata and create train/val/test splits.

This gives a clean dataset for both imitation learning and supervised learning baselines, while preserving trajectory structure for sequential analysis.

## Repository structure

- `src/nav_dataset/` – source code (dataset, models, training, audits, visualization)
- `data/maze_small/` – retained small-maze dataset
- `models/torch/` – saved PyTorch models
- `models/sklearn/` – saved baseline sklearn models
- `artifacts/figures/` – training curves, KPI comparison, visualizations
- `artifacts/reports/` – metrics summaries and audit reports
- `artifacts/inference/` – extracted inference outputs and run logs copy
- `logs/` – full training logs
- `docs/` – project documentation
- `configs/`, `scripts/`, `tests/` – standard ML repo placeholders

## Quick start

Install dependencies:

- Python 3.10+
- `numpy`

Generate a dataset:

- `python -m src.nav_dataset.generate_dataset --output data/maze_small --episodes 700 --map-type maze --height 49 --width 49 --min-start-goal-l2 14 --seed 42`

Generated files:

- `data/maze_small/navigation_dataset.npz` – arrays for ML training
- `data/maze_small/metadata.json` – generation settings and summary statistics
- `data/maze_small/episode_data.npz` – full occupancy grid per episode and endpoints
- `data/maze_small/expert_trajectories.jsonl` – complete grid-based path coordinates per episode

Train Torch model (100 epochs) and produce artifacts:

- `python -m src.nav_dataset.train_torch_maze --dataset data/maze_small/navigation_dataset.npz --episode-data data/maze_small/episode_data.npz --trajectories data/maze_small/expert_trajectories.jsonl --output models/torch/run_small --artifacts artifacts/figures --epochs 100 --seed 42 --action-loss hybrid --use-class-weights`

Audit dataset quality and distribution:

- `python -m src.nav_dataset.audit_dataset --dataset data/maze_small/navigation_dataset.npz --output artifacts/reports/data_audit_small`

## Data format (`navigation_dataset.npz`)

- `local_patch` : `(N, P, P)` uint8
- `relative_goal` : `(N, 2)` float32
- `agent_position` : `(N, 2)` int64 grid coordinates `[row, col]`
- `action_id` : `(N,)` int64
- `cost_to_goal` : `(N,)` float32
- `episode_id` : `(N,)` int64
- `timestep` : `(N,)` int64
- `split` : `(N,)` int8 where `0=train, 1=val, 2=test`

## Episode-level artifacts for full path learning

- `episode_data.npz`
	- `episode_grid` : `(E, H, W)` uint8
	- `episode_start` : `(E, 2)` int64
	- `episode_goal` : `(E, 2)` int64
	- `episode_split` : `(E,)` int8
- `expert_trajectories.jsonl`
	- One JSON object per episode with `start`, `goal`, and complete `path` as grid coordinates `[[r0,c0], [r1,c1], ...]`

This is the representation you can use directly for **grid-based complete-path learning**.

For a formal write-up of the dataset rationale and collection process, see `docs/2_1_dataset_description.md`.

## Logs and inference outputs

- Full run logs are kept in `logs/`
- Inference snippets and copied run logs are kept in `artifacts/inference/`

## KPI compatibility

The data includes what you need to evaluate your proposed KPIs downstream:

- **Success rate**: replay learned policy in held-out maps
- **Optimality ratio**: compare policy trajectory length to A\* path length
- **Computation time per decision**: benchmark model inference vs A\* query time

---

If you want, I can also add the next step: an evaluation script that directly computes success rate / optimality ratio / decision latency for A\* vs learned policies.


some files were created using Claude code, code reviewed by Surya 