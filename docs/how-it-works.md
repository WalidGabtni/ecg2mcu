# How ecg2mcu works

```
PTB-XL / sample ──► normalize ──► train CNN1D (PyTorch, float32)
                                        │
                      PyTorch weights ──► equivalent Keras model ──► TFLite INT8 (post-training)
                                        │                                   │
                                        │                         model.tflite + C header
                                        ▼                                   ▼
                              float predictions  ◄── compare ──►  INT8 predictions  ──► verify_report
```

## Pipeline stages

1. **Data.** Windows are 10 s at 100 Hz (1000 samples). PTB-XL uses its recommended folds: 1-8 train, 9 validation, 10 test. Labels are the five diagnostic superclasses (NORM, MI, STTC, CD, HYP), multi-label.
2. **Training.** A small 1D-CNN (four conv blocks of 32/64/128/256 filters, global average pooling, two dense layers) trained with binary cross-entropy. The checkpoint with the lowest validation loss is kept.
3. **Export.** The PyTorch weights are copied into an equivalent Keras model (the export refuses to continue if the outputs differ by more than 1e-3), then converted with TFLite's full-integer post-training quantization. The calibration set is a slice of the training windows.
4. **Header.** The `.tflite` bytes are written as an 8-byte aligned `PROGMEM` array that the TFLite Micro firmware includes directly.
5. **Verify.** The INT8 model is run on the test split and compared with the float model.

## What the verification checks

| Check | Default limit | Why |
|---|---|---|
| Macro AUC drop (float - INT8) | 0.01 | The quantized model must rank cases like the float one |
| Mean probability difference | 0.05 | Catches models that rank similarly but are badly calibrated |
| Float logits outside the INT8 output range | 5% | A saturated output tensor clips predictions; this is the failure that went unnoticed in a real export |

The command exits with a non-zero status when a check fails, so it can gate a CI pipeline.

## Normalization must match the firmware

The firmware z-scores each acquired window before quantizing it. `data.normalization = "window"` (the default) does the same for every lead, so training and deployment agree. `"global"` reproduces the original thesis code, which normalized with statistics pooled over all 12 leads; on Lead I this differs from the firmware and cost about 0.01 macro AUC on the float model.

## Memory estimate

`verify_report.md` includes a rough budget for the device: flash is the model size, and the tensor arena lower bound is the largest sum of a layer's input and output activations (INT8). A 1.5x factor gives the recommended arena. The factor was calibrated against one measurement (the thesis model: 48 KB lower bound, 68 KB measured), so treat the number as a starting point and confirm that `AllocateTensors()` succeeds on the board.

## Limitations

- One architecture: the 1D-CNN. GRU layers cannot go through PyTorch's QAT path and are not supported.
- Post-training quantization only. Quantization-aware training is a possible future option.
- The bundled sample is for smoke tests; its metrics are not meaningful.
- Research code, not a medical device.
