import numpy as np
from typing import Optional
import torch.nn as nn

def cosine_scheduler(
        t: int, 
        T: int, 
        lr_max: Optional[float] = 1, 
        lr_min: Optional[float] = 0.996) -> float:
    """
    Cosine learning rate scheduler.
    On BYOL, it is used to update the learning rate based on the current step as such:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(t / T * pi))
        Based on https://arxiv.org/pdf/2006.07733.pdf, page 5.

    Args:
        t (int): Current step.
        T (int): Maximum steps.
        lr_max (float): Maximum learning rate.
        lr_min (float): Minimum learning rate.

    Returns:
        float: The learning rate at the current step.
    """
    if T == 1:
        return lr_max
    if t == T:
        return lr_min
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(t / T * np.pi))

def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """
    Update the momentum of the EMA model.
    On BYOL, it updates the target network based on the online network's weights as such:
        target_network = m * target_network + (1 - m) * online_network
        Based on https://arxiv.org/pdf/2006.07733.pdf, page 4.

    Args:
        model (nn.Module): The online network.
        model_ema (nn.Module): The target network.
        m (float): The momentum.
    """
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data = param_ema.data * m + param.data * (1.0 - m)

def get_default_byol_hparams():
    return {
        "max_epochs": 10,
        "input_dim": 512,
        "hidden_dim": 1024,
        "projection_dim": 256,
    }