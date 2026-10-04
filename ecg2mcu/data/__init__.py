"""Dataset access: the bundled sample or the full PTB-XL dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from .preprocess import zscore_window

SAMPLE_FILE = Path(__file__).with_name("sample_ptbxl_lead1.npz")


@dataclass
class Splits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def in_channels(self) -> int:
        return self.x_train.shape[1]


def load_sample(normalization: str = "window") -> dict:
    """Load the bundled Lead I sample (a small subset of PTB-XL, CC BY 4.0).

    It is meant for smoke tests and demos: the metrics are not meaningful.
    """
    if normalization != "window":
        raise ValueError("The bundled sample contains Lead I only, so only 'window' normalization is available.")
    with np.load(SAMPLE_FILE) as z:
        return {s: (zscore_window(z[f"x_{s}"]), z[f"y_{s}"]) for s in ("train", "val", "test")}


def load_splits(cfg: Config) -> Splits:
    if cfg.data.source == "sample":
        if cfg.lead_indices != [0]:
            raise ValueError("The bundled sample has Lead I only: set data.leads = 'lead1'.")
        parts = load_sample(cfg.data.normalization)
    else:
        from .ptbxl import load_ptbxl

        parts = load_ptbxl(cfg.data.path, cfg.lead_indices, cfg.data.normalization, cfg.data.cache_dir)
    return Splits(*parts["train"], *parts["val"], *parts["test"])
