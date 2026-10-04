import numpy as np

from ecg2mcu.config import load_config
from ecg2mcu.data import load_splits


def test_bundled_sample_shapes_and_normalization():
    s = load_splits(load_config())
    assert s.x_train.shape == (300, 1, 1000) and s.y_train.shape == (300, 5)
    assert s.x_val.shape[0] == 100 and s.x_test.shape[0] == 150
    assert s.x_train.dtype == np.float32
    assert np.allclose(s.x_train.mean(axis=-1), 0, atol=1e-3)
    assert s.y_test.sum(axis=0).min() > 0   # every class present in the test split
