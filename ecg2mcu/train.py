"""Float32 training of the 1D-CNN."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data import Splits
from .metrics import macro_auc, sigmoid
from .models import CNN1D, count_parameters


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def build_model(cfg: Config, in_channels: int) -> CNN1D:
    return CNN1D(in_channels=in_channels, fc_size=cfg.model.fc_size, dropout=cfg.model.dropout)


def train(cfg: Config, splits: Splits, out_dir: str | Path, log=print) -> tuple[CNN1D, dict]:
    """Train with BCE loss, keep the checkpoint with the lowest validation loss."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = cfg.train
    set_seed(t.seed)
    device = resolve_device(t.device)

    model = build_model(cfg, splits.in_channels).to(device)
    log(f"[train] device={device} channels={splits.in_channels} params={count_parameters(model):,} "
        f"train={len(splits.x_train)} val={len(splits.x_val)} test={len(splits.x_test)}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=t.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    x_train = torch.from_numpy(splits.x_train)
    y_train = torch.from_numpy(splits.y_train)
    x_val = torch.from_numpy(splits.x_val).to(device)
    y_val = torch.from_numpy(splits.y_val).to(device)

    best_val, best_state, history = float("inf"), None, []
    for epoch in range(1, t.epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        total = 0.0
        for i in range(0, len(order), t.batch_size):
            idx = order[i:i + t.batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        train_loss = total / len(order)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"[train] epoch {epoch:>3}/{t.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.to("cpu").eval()
    torch.save(best_state, out_dir / "model.pth")

    test_probs = sigmoid(predict_logits(model, splits.x_test, torch.device("cpu")))
    summary = {
        "best_val_loss": best_val,
        "test_macro_auc_float": macro_auc(splits.y_test, test_probs),
        "epochs": t.epochs,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[train] float32 test macro AUC = {summary['test_macro_auc_float']:.4f}  -> {out_dir / 'model.pth'}")
    return model, summary
