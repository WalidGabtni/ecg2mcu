# Firmware template

ESP32 sketch (`ecg_inference/`) that acquires a 10 s Lead I window at 100 Hz from an AD8232, z-normalizes it, runs the exported INT8 model with TensorFlow Lite Micro, shows the result on an ST7735 display and can post vitals to Supabase.

1. `ecg2mcu export --config <config> --firmware-dir firmware/ecg_inference` writes `cnn1d_model.h` next to the sketch.
2. Copy `secrets.example.h` to `secrets.h` and fill in your values (`secrets.h` is gitignored).
3. Board: ESP32 Dev Module, Partition Scheme: Huge APP (3 MB, no OTA).

| Signal | GPIO |
|--------|------|
| TFT CS / RST / DC | 5 / 4 / 15 |
| AD8232 output | 34 |
| AD8232 LO+ / LO- / SDN | 32 / 33 / 27 |
| MAX30100 SDA / SCL | 21 / 22 |

Set `TENSOR_ARENA_SIZE` in the sketch to at least the recommended arena in `verify_report.md`. The firmware normalizes each window on its own, so train with `data.normalization = "window"`.
