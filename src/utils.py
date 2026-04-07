from __future__ import annotations

import random
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_device(require_cuda: bool = True) -> torch.device:
    has_cuda = torch.cuda.is_available()
    if require_cuda and not has_cuda:
        raise RuntimeError(
            "CUDA is required for this project run but no CUDA device is available. "
            "Install a CUDA-enabled PyTorch build and ensure an NVIDIA GPU is visible."
        )
    device = torch.device("cuda" if has_cuda else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start
        return False
