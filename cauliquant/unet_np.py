"""Pieces of the U-Net code that do not need PyTorch."""
import numpy as np


def normalise(gray):
    g = gray.astype(np.float32) / 255.0
    return (g - g.mean()) / (g.std() + 1e-6)
