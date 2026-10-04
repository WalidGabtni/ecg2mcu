"""Command line interface: ecg2mcu train | export | verify | run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .config import Config, load_config
from .data import load_splits
from .models import CNN1D
from .train import build_model, train


def _load_model(cfg: Config, run_dir: Path, in_channels: int) -> CNN1D:
    ckpt = run_dir / "model.pth"
    if not ckpt.exists():
        raise SystemExit(f"No trained model at {ckpt}. Run `ecg2mcu train` first.")
    model = build_model(cfg, in_channels)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def _cmd_train(cfg: Config, args) -> int:
    train(cfg, load_splits(cfg), cfg.export.out_dir)
    return 0


def _cmd_export(cfg: Config, args) -> int:
    from .export import export_model

    splits = load_splits(cfg)
    model = _load_model(cfg, Path(cfg.export.out_dir), splits.in_channels)
    n = min(cfg.export.calibration_samples, len(splits.x_train))
    export_model(model, splits.x_train[:n], cfg.export.out_dir, cfg.export.var_name, cfg.export.firmware_dir)
    return 0


def _cmd_verify(cfg: Config, args) -> int:
    from .verify import verify

    run_dir = Path(cfg.export.out_dir)
    splits = load_splits(cfg)
    model = _load_model(cfg, run_dir, splits.in_channels)
    tflite = run_dir / "model.tflite"
    if not tflite.exists():
        raise SystemExit(f"No exported model at {tflite}. Run `ecg2mcu export` first.")
    report = verify(model, tflite, splits.x_test, splits.y_test, cfg, run_dir)
    print((run_dir / "verify_report.md").read_text())
    return 0 if report.passed else 1


def _cmd_run(cfg: Config, args) -> int:
    for step in (_cmd_train, _cmd_export, _cmd_verify):
        code = step(cfg, args)
        if code:
            return code
    return 0


COMMANDS = {"train": _cmd_train, "export": _cmd_export, "verify": _cmd_verify, "run": _cmd_run}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ecg2mcu", description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", "-c", help="TOML config file (see configs/)")
    parser.add_argument("--out-dir", help="override export.out_dir")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--data-path", help="override data.path (PTB-XL directory)")
    parser.add_argument("--firmware-dir", help="also copy the C header to this firmware folder")
    args = parser.parse_args(argv)

    overrides = {}
    if args.out_dir:
        overrides["export.out_dir"] = args.out_dir
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.data_path:
        overrides["data.path"] = args.data_path
    if args.firmware_dir:
        overrides["export.firmware_dir"] = args.firmware_dir
    cfg = load_config(args.config, overrides)
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
