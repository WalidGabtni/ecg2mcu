import numpy as np

from ecg2mcu.data.preprocess import zscore_global, zscore_window


def test_window_zscore_is_per_lead():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((4, 3, 1000)) * [[[1], [5], [20]]] + [[[0], [3], [-7]]]).astype(np.float32)
    z = zscore_window(x)
    assert z.dtype == np.float32
    assert np.allclose(z.mean(axis=-1), 0, atol=1e-4)
    assert np.allclose(z.std(axis=-1), 1, atol=1e-3)


def test_global_zscore_keeps_relative_lead_amplitudes():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 12, 1000)).astype(np.float32)
    x[:, 0] *= 10
    z = zscore_global(x)
    assert np.allclose(z.mean(axis=(1, 2)), 0, atol=1e-4)
    assert z[:, 0].std() > 3 * z[:, 5].std()
