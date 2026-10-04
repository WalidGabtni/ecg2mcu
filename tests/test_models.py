import torch

from ecg2mcu.models import CNN1D, count_parameters


def test_output_shape_for_lead_counts():
    for channels in (1, 6, 12):
        out = CNN1D(in_channels=channels)(torch.randn(2, channels, 1000))
        assert out.shape == (2, 5)


def test_state_dict_keys_match_thesis_checkpoints():
    keys = set(CNN1D().state_dict())
    for name in ("conv1.weight", "bn1.running_var", "conv4.weight", "bn4.bias", "fc1.weight", "fc2.bias"):
        assert name in keys
    assert not any(k.startswith(("quant", "dequant")) for k in keys)


def test_parameter_count_lead1():
    assert count_parameters(CNN1D(in_channels=1)) == 201_381
