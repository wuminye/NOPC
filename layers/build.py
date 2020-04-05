import torch
from .perceptual_loss import Perceptual_loss


def make_loss():
    return Perceptual_loss()
