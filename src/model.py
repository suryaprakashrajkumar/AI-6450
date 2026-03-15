from __future__ import annotations

import torch
from torch import nn


class PathPolicyNet(nn.Module):
    def __init__(self, hidden_dim: int = 256, n_actions: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(3200 + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions),
        )

    def forward(self, grid: torch.Tensor, pos: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        grid_feat = self.encoder(grid)
        x = torch.cat([grid_feat, pos, goal], dim=1)
        return self.head(x)


class PathPolicyNetLarge(nn.Module):
    def __init__(self, hidden_dim: int = 512, n_actions: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(6400 + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, grid: torch.Tensor, pos: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        grid_feat = self.encoder(grid)
        x = torch.cat([grid_feat, pos, goal], dim=1)
        return self.head(x)


class PathPolicyMLP(nn.Module):
    def __init__(self, hidden_dim: int = 256, n_actions: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100 + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, grid: torch.Tensor, pos: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        flat_grid = grid.flatten(start_dim=1)
        x = torch.cat([flat_grid, pos, goal], dim=1)
        return self.net(x)


def build_model(model_type: str, hidden_dim: int, n_actions: int = 8) -> nn.Module:
    if model_type == "cnn_small":
        return PathPolicyNet(hidden_dim=hidden_dim, n_actions=n_actions)
    if model_type == "cnn_large":
        return PathPolicyNetLarge(hidden_dim=hidden_dim, n_actions=n_actions)
    if model_type == "mlp":
        return PathPolicyMLP(hidden_dim=hidden_dim, n_actions=n_actions)
    raise ValueError(f"Unknown model_type '{model_type}'. Supported: cnn_small, cnn_large, mlp")
