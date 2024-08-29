import random
import numpy as np
import torch

def set_seed(seed: int, device_specific: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)