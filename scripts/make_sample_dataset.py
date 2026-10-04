"""Build the small Lead I sample bundled with ecg2mcu from a local PTB-XL copy.

Usage: python scripts/make_sample_dataset.py --ptbxl <path to PTB-XL 1.0.3>

Picks a class-balanced subset (train from folds 1-8, validation from fold 9, test from fold 10)
and stores the raw Lead I signal at 100 Hz as float16. PTB-XL is released under CC BY 4.0:
Wagner et al., https://physionet.org/content/ptb-xl/1.0.3/
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SPLITS = {"train": (range(1, 9), 300), "val": ([9], 100), "test": ([10], 150)}


def labels(codes, statements):
    y = np.zeros(len(CLASSES), dtype=np.float32)
    for code in codes:
        if code in statements.index:
            sc = statements.loc[code, "diagnostic_class"]
            if isinstance(sc, str) and sc in CLASSES:
                y[CLASSES.index(sc)] = 1
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "ecg2mcu" / "data" / "sample_ptbxl_lead1.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.ptbxl)
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    statements = statements[statements["diagnostic"] == 1]
    df["y"] = df["scp_codes"].apply(lambda c: labels(c, statements))
    df = df[df["y"].apply(lambda v: v.sum() > 0)]

    rng = np.random.default_rng(args.seed)
    out = {}
    for split, (folds, size) in SPLITS.items():
        pool = df[df["strat_fold"].isin(list(folds))]
        chosen = []
        for c in range(len(CLASSES)):
            ids = pool.index[pool["y"].apply(lambda v: v[c] == 1)].to_numpy()
            chosen += list(rng.choice(ids, size=size // len(CLASSES), replace=False))
        chosen = list(dict.fromkeys(chosen))
        rest = [i for i in pool.index if i not in set(chosen)]
        chosen += list(rng.choice(rest, size=size - len(chosen), replace=False))
        rng.shuffle(chosen)

        xs, ys = [], []
        for ecg_id in chosen:
            sig, _ = wfdb.rdsamp(str(root / pool.loc[ecg_id, "filename_lr"]))
            xs.append(sig[:1000, 0].astype(np.float16))   # Lead I
            ys.append(pool.loc[ecg_id, "y"])
        out[f"x_{split}"] = np.stack(xs)[:, None, :]
        out[f"y_{split}"] = np.stack(ys)
        print(split, out[f"x_{split}"].shape, "positives per class:", out[f"y_{split}"].sum(axis=0).astype(int))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, f"{Path(args.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
