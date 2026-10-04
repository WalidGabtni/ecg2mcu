import pytest

from ecg2mcu.config import load_config


def test_defaults():
    cfg = load_config()
    assert cfg.lead_indices == [0]
    assert cfg.in_channels == 1
    assert cfg.data.normalization == "window"


def test_presets_and_overrides():
    cfg = load_config(overrides={"data.leads": "limb6", "train.epochs": 3})
    assert cfg.lead_indices == [0, 1, 2, 3, 4, 5]
    assert cfg.train.epochs == 3


def test_toml_file(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[data]\nleads = [0, 2]\n[model]\nfc_size = 128\n')
    cfg = load_config(p)
    assert cfg.lead_indices == [0, 2] and cfg.model.fc_size == 128


@pytest.mark.parametrize("text", [
    '[data]\nleads = "nope"\n',
    '[data]\nleads = [12]\n',
    '[data]\nnormalization = "other"\n',
    '[bogus]\nx = 1\n',
    '[train]\nunknown_key = 1\n',
])
def test_invalid_configs_are_rejected(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    with pytest.raises(ValueError):
        load_config(p)


def test_shipped_configs_load():
    from pathlib import Path
    for path in (Path(__file__).parents[1] / "configs").glob("*.toml"):
        load_config(path)
