"""Run configuration, loaded from a TOML file.

Every key has a default, so a config file only needs the values it changes.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# PTB-XL record order: I, II, III, aVR, aVL, aVF, V1..V6
LEAD_PRESETS = {
    "lead1": [0],
    "lead2": [1],
    "lead3": [2],
    "limb6": [0, 1, 2, 3, 4, 5],
    "all12": list(range(12)),
}


@dataclass
class DataConfig:
    source: str = "sample"        # "sample" (bundled) or "ptbxl" (full dataset)
    path: str = ""                # PTB-XL directory (source = "ptbxl")
    leads: str | list[int] = "lead1"
    normalization: str = "window"  # "window": z-score each lead per window (matches the firmware)
                                   # "global": z-score with all-lead statistics (original thesis code)
    cache_dir: str = ""           # optional: cache parsed PTB-XL arrays as .npz


@dataclass
class ModelConfig:
    fc_size: int = 256
    dropout: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    device: str = "auto"          # "auto", "cpu" or "cuda"


@dataclass
class ExportConfig:
    out_dir: str = "runs/default"
    calibration_samples: int = 200
    var_name: str = "cnn1d_model"
    firmware_dir: str = ""        # if set, the C header is also copied here
    tolerance_auc: float = 0.01   # verification fails if INT8 AUC drops more than this
    tolerance_prob: float = 0.05  # ... or if the mean probability difference exceeds this


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @property
    def lead_indices(self) -> list[int]:
        leads = self.data.leads
        if isinstance(leads, str):
            if leads not in LEAD_PRESETS:
                raise ValueError(f"Unknown lead preset '{leads}'. Use one of {list(LEAD_PRESETS)} or a list of indices.")
            return list(LEAD_PRESETS[leads])
        if not leads or any(not 0 <= i < 12 for i in leads):
            raise ValueError("leads must be indices between 0 and 11")
        return list(leads)

    @property
    def in_channels(self) -> int:
        return len(self.lead_indices)


def _merge(obj: Any, values: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{path}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{path}{key}' must be a table")
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a TOML config. `overrides` uses dotted keys, e.g. {"train.epochs": 5}."""
    cfg = Config()
    if path:
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        _merge(cfg, {section: {key: value}})
    if cfg.data.normalization not in ("window", "global"):
        raise ValueError("data.normalization must be 'window' or 'global'")
    if cfg.data.source not in ("sample", "ptbxl"):
        raise ValueError("data.source must be 'sample' or 'ptbxl'")
    cfg.lead_indices  # validate
    return cfg
