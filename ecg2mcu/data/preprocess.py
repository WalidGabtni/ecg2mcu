"""Input normalization.

The firmware normalizes each acquired window before quantizing it, so training
must use the same preprocessing. Mismatched normalization silently costs
accuracy on the device, which is why "window" is the default.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def zscore_window(x: np.ndarray) -> np.ndarray:
    """Z-score every lead of every window independently. x has shape (N, C, T)."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / (std + EPS)).astype(np.float32)


def zscore_global(x_all_leads: np.ndarray) -> np.ndarray:
    """Z-score a window with statistics pooled over all its leads (the original thesis code).

    x_all_leads has shape (N, 12, T); statistics are computed over axes (1, 2).
    """
    mean = x_all_leads.mean(axis=(1, 2), keepdims=True)
    std = x_all_leads.std(axis=(1, 2), keepdims=True)
    return ((x_all_leads - mean) / (std + EPS)).astype(np.float32)
