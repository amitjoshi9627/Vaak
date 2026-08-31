import os
import random

import numpy as np
import torch


def get_optimal_device() -> torch.device:
    """Determine the optimal available PyTorch compute device.

    Returns:
        torch.device configured for CUDA, MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set global seed across random, NumPy, and PyTorch backends."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
