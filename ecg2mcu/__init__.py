"""ecg2mcu: train, quantize and verify ECG classifiers for microcontrollers."""

__version__ = "0.1.0"

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
SAMPLE_RATE_HZ = 100
WINDOW_SAMPLES = 1000
