"""End to end on the bundled sample: train, export to INT8 TFLite, verify."""

import pytest

pytest.importorskip("tensorflow")

from ecg2mcu.config import load_config
from ecg2mcu.data import load_splits
from ecg2mcu.export import export_model
from ecg2mcu.train import train
from ecg2mcu.verify import verify


@pytest.mark.slow
def test_train_export_verify(tmp_path):
    cfg = load_config(overrides={"train.epochs": 8, "export.out_dir": str(tmp_path),
                                 "export.tolerance_auc": 0.05})
    splits = load_splits(cfg)
    model, summary = train(cfg, splits, tmp_path, log=lambda *_: None)
    assert (tmp_path / "model.pth").exists()

    info = export_model(model, splits.x_train[:100], tmp_path, "cnn1d_model", log=lambda *_: None)
    assert info["weight_transfer_max_diff"] < 1e-3
    assert (tmp_path / "model.tflite").stat().st_size > 50_000
    assert "alignas(8)" in (tmp_path / "cnn1d_model.h").read_text()

    report = verify(model, tmp_path / "model.tflite", splits.x_test, splits.y_test, cfg, tmp_path)
    assert report.passed, report.reasons
    assert (tmp_path / "verify_report.md").exists()
