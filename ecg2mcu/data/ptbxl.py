"""PTB-XL loader (100 Hz records, diagnostic superclasses).

Splits follow the dataset's recommended stratified folds: 1-8 train, 9 validation, 10 test.
Download the dataset with `python scripts/download_ptbxl.py`.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np

from .. import CLASS_NAMES, WINDOW_SAMPLES

SPLIT_FOLDS = {"train": range(1, 9), "val": [9], "test": [10]}


def _require_deps():
    try:
        import pandas as pd
        import wfdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading PTB-XL needs pandas and wfdb: pip install 'ecg2mcu[ptbxl]'"
        ) from exc
    return pd, wfdb


def _labels_for(scp_codes: dict, statements) -> np.ndarray:
    vec = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for code in scp_codes:
        if code in statements.index:
            superclass = statements.loc[code, "diagnostic_class"]
            if isinstance(superclass, str) and superclass in CLASS_NAMES:
                vec[CLASS_NAMES.index(superclass)] = 1.0
    return vec


def _load_fold_group(root: Path, folds, cache_dir: str, normalization: str):
    """Return (x_all_leads (N,12,T) raw signals, y (N,5)) for the given folds."""
    pd, wfdb = _require_deps()
    key = hashlib.md5(f"{root.resolve()}|{list(folds)}".encode()).hexdigest()[:10]
    cache_file = Path(cache_dir) / f"ptbxl_{key}.npz" if cache_dir else None
    if cache_file and cache_file.exists():
        with np.load(cache_file) as z:
            return z["x"], z["y"]

    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    statements = statements[statements["diagnostic"] == 1]

    df = df[df["strat_fold"].isin(list(folds))]
    xs, ys = [], []
    for _, row in df.iterrows():
        y = _labels_for(row["scp_codes"], statements)
        if y.sum() == 0:
            continue
        signal, _ = wfdb.rdsamp(str(root / row["filename_lr"]))
        xs.append(signal.T.astype(np.float32)[:, :WINDOW_SAMPLES])
        ys.append(y)
    x, y = np.stack(xs), np.stack(ys)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, x=x, y=y)
    return x, y


def load_ptbxl(path: str, leads: list[int], normalization: str, cache_dir: str = ""):
    """Return {"train": (x, y), "val": (x, y), "test": (x, y)}; x has shape (N, len(leads), 1000)."""
    from .preprocess import zscore_global, zscore_window

    root = Path(path)
    if not (root / "ptbxl_database.csv").exists():
        raise FileNotFoundError(
            f"PTB-XL not found at '{path}'. Run `python scripts/download_ptbxl.py` and set data.path."
        )
    out = {}
    for split, folds in SPLIT_FOLDS.items():
        x12, y = _load_fold_group(root, folds, cache_dir, normalization)
        if normalization == "global":
            x = zscore_global(x12)[:, leads, :]
        else:
            x = zscore_window(x12[:, leads, :])
        out[split] = (x, y)
    return out
