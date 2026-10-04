# ecg2mcu

Train an ECG classifier, quantize it to INT8, and **verify that the quantized model still works** before it goes onto an ESP32. One command takes a dataset to a TFLite Micro model, a C header for your firmware, and a report that tells you whether the export is trustworthy.

It grew out of a Master's thesis on running cardiac classification directly on a microcontroller ([ECG-cardiac-monitor](https://github.com/WalidGabtni/ECG-cardiac-monitor)), where a quantized model that looked fine had in fact collapsed to chance level. The verification step exists so that cannot happen silently again.

## Quickstart

```bash
git clone https://github.com/WalidGabtni/ecg2mcu.git
cd ecg2mcu
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                  # needs Python 3.11+

ecg2mcu run --config configs/sample_lead1.toml           # about a minute on a CPU
```

This trains on a small bundled sample of PTB-XL (no download), exports an INT8 model and prints a verification report. The sample's metrics are not meaningful; it proves that the pipeline works on your machine.

```
# ecg2mcu verification: PASSED

| Float32 macro AUC | ... |
| INT8 macro AUC | ... |
| Mean / max probability difference | ... |
| Flash (model) | 226.6 KB |
| Tensor arena (lower bound / recommended) | 46.9 KB / 72 KB |
```

Outputs land in `runs/<name>/`: `model.pth`, `model.tflite`, `cnn1d_model.h`, `verify_report.md` and `verify_report.json`.

## Train on the full PTB-XL dataset

```bash
python scripts/download_ptbxl.py                         # ~1.8 GB from PhysioNet
ecg2mcu run --config configs/ptbxl_lead1.toml            # Lead I, the single-lead prototype
```

Other configs: `ptbxl_limb6.toml` (six frontal leads) and `ptbxl_all12.toml`. Run the stages separately with `ecg2mcu train | export | verify`.

## Why the verification step matters

A quantized model can fail without any error. In the thesis project, the Lead I model flashed on the device scored **0.529 macro AUC** (chance level) while its float version scored **0.8385**: the INT8 output tensor could only represent logits from -0.98 to 0, while the real logits range from -17 to +14. `ecg2mcu verify` reports exactly this:

```
OLD export:  FAILED  float AUC 0.8385, INT8 AUC 0.5289, mean probability difference 0.264
  - INT8 macro AUC dropped by 0.3096 (limit 0.01)
  - mean probability difference 0.2642 exceeds 0.05
  - 87% of float logits fall outside the INT8 output range [-0.98, 0.00]

NEW export:  PASSED  float AUC 0.8385, INT8 AUC 0.8387, mean probability difference 0.005
```

The command exits with a non-zero status on failure, so it fits in CI. Details are in [docs/how-it-works.md](docs/how-it-works.md).

## Verified exports

PTB-XL test fold (fold 10), the thesis 1D-CNN checkpoints:

| Input | Float32 macro AUC | INT8 `.tflite` macro AUC | INT8 size |
|-------|:-----------------:|:------------------------:|:---------:|
| Lead I | 0.8385 | 0.8387 | 227 KB |
| 6 frontal leads | 0.8873 | 0.8874 | 228 KB |

## Use the model in firmware

`firmware/ecg_inference/` is an ESP32 sketch (Arduino framework) that acquires a 10 s window from an AD8232 ECG front-end, normalizes it, runs the model with TensorFlow Lite Micro, shows the result on an ST7735 display, and optionally posts vitals to Supabase.

```bash
ecg2mcu export --config configs/ptbxl_lead1.toml --firmware-dir firmware/ecg_inference
cp firmware/ecg_inference/secrets.example.h firmware/ecg_inference/secrets.h   # add your Wi-Fi and Supabase details
```

Then open the sketch in the Arduino IDE (board: ESP32 Dev Module, partition scheme: Huge APP) and upload. It needs the libraries `Adafruit GFX`, `Adafruit ST7735 and ST7789`, `ArduinoJson` (v7), `Chirale_TensorFlowLite` and `MAX30100lib`. Check that the sketch's `TENSOR_ARENA_SIZE` is at least the recommended arena from the report. The firmware compiles; hardware validation of this toolkit's exports is still pending.

## Configuration

Every option has a default; a config file only lists what it changes. The main ones:

| Key | Meaning |
|-----|---------|
| `data.source` | `sample` (bundled) or `ptbxl` |
| `data.leads` | `lead1`, `lead2`, `lead3`, `limb6`, `all12`, or a list of indices |
| `data.normalization` | `window` (matches the firmware) or `global` (original thesis code) |
| `train.epochs`, `train.lr`, `train.seed` | training settings |
| `export.tolerance_auc`, `export.tolerance_prob` | limits for the verification |
| `export.firmware_dir` | also copy the C header into a firmware folder |

## Scope and limitations

- One architecture: a small 1D-CNN. GRU layers cannot go through PyTorch's QAT path and are not supported.
- Post-training INT8 quantization; quantization-aware training is a possible future option.
- Research code. Not a medical device and not for diagnosis.

## Roadmap

- A second dataset (MIT-BIH) and a Colab notebook
- Optional quantization-aware training
- Benchmarks across ESP32 variants

Contributions and issues are welcome.

## License

The code is released under the [MIT License](LICENSE): anyone can use, modify and redistribute it. The bundled sample data is a subset of PTB-XL and keeps its own license (CC BY 4.0), see below.

## Data and citation

The bundled sample is a subset of PTB-XL, released under CC BY 4.0. Wagner, P. et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154 (2020). https://physionet.org/content/ptb-xl/1.0.3/

## Author

**Walid Gabtni**: International Research Master's in Cyber-Physical Systems, ISSAT Mateur. [@WalidGabtni](https://github.com/WalidGabtni)
