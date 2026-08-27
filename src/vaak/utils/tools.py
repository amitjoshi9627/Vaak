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
