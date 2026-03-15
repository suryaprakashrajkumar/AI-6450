from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Config:
    grid_size: int = 10
    n_actions: int = 8
    seed: int = 42

    min_obstacle_density: float = 0.10
    max_obstacle_density: float = 0.40

    n_maps: int = 700
    pairs_per_map: int = 8
    min_astar_path_len: int = 4
    density_bins: int = 6
    rebalance_actions: bool = True
    max_action_ratio: float = 2.5

    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15

    hidden_dim: int = 256
    model_type: str = "cnn_small"
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 25
    early_stop_patience: int = 5
    label_smoothing: float = 0.05

    rollout_step_cap: int = 60
    enable_astar_fallback: bool = True
    enable_confidence_fallback: bool = True
    fallback_confidence_threshold: float = 0.38
    fallback_min_step: int = 6
    fallback_no_progress_patience: int = 2
    fallback_max_calls: int = 1
    qa_action_imbalance_ratio_threshold: float = 0.15


def load_config(config_path: str | None = None) -> Config:
    if config_path is None:
        return Config()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return Config(**payload)


def save_config(config: Config, output_path: str) -> None:
    Path(output_path).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
