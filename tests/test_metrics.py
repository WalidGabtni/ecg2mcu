import numpy as np

from ecg2mcu.metrics import macro_auc, per_class_auc, sigmoid


def test_perfect_and_undefined_classes():
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=np.float32)
    p = np.array([[0.9, 0.2], [0.1, 0.3], [0.8, 0.1], [0.2, 0.4]])
    per = per_class_auc(y, p)
    assert per[0] == 1.0 and np.isnan(per[1])   # class 1 has no positives
    assert macro_auc(y, p) == 1.0


def test_sigmoid_is_stable_and_monotonic():
    z = np.array([-20.0, 0.0, 20.0])
    s = sigmoid(z)
    assert s[0] < 1e-6 and s[1] == 0.5 and s[2] > 1 - 1e-6
