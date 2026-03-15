import torch

from src.model import PathPolicyNet, build_model


def test_model_shape():
    model = PathPolicyNet()
    grid = torch.randn(4, 1, 10, 10)
    pos = torch.randn(4, 2)
    goal = torch.randn(4, 2)
    out = model(grid, pos, goal)
    assert out.shape == (4, 8)


def test_model_factory_variants_shape():
    grid = torch.randn(2, 1, 10, 10)
    pos = torch.randn(2, 2)
    goal = torch.randn(2, 2)

    for model_type, hidden_dim in [("mlp", 256), ("cnn_small", 256), ("cnn_large", 512)]:
        model = build_model(model_type, hidden_dim=hidden_dim, n_actions=8)
        out = model(grid, pos, goal)
        assert out.shape == (2, 8)
