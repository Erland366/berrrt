import torch


def get_default_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seed(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
