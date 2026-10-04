"""Check that the quantized model still behaves like the float model.

A quantized export can go wrong silently (bad calibration, a saturated output range, a
mismatched converter). This compares float32 and INT8 predictions on the held-out test
split and fails loudly if they diverge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .export import estimate_memory
from .metrics import macro_auc, per_class_auc, sigmoid
from .models import CNN1D
from .train import predict_logits


def _interpreter(model_path: str):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def run_tflite(model_path: str | Path, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run an INT8 model on windows x (N, channels, T); returns (logits, tensor info)."""
    interp = _interpreter(str(model_path))
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    logits = np.empty((len(x), out["shape"][-1]), dtype=np.float32)
    for i in range(len(x)):
        window = x[i].T[None, :, :]                      # (1, T, channels)
        q = np.clip(np.round(window / s_in + z_in), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        logits[i] = (interp.get_tensor(out["index"])[0].astype(np.float32) - z_out) * s_out
    info = {
        "input_scale": float(s_in), "input_zero_point": int(z_in),
        "output_scale": float(s_out), "output_zero_point": int(z_out),
        # Logits outside this range are clipped by the INT8 output tensor.
        "output_logit_range": [float((-128 - z_out) * s_out), float((127 - z_out) * s_out)],
    }
    return logits, info


@dataclass
class VerifyReport:
    float_macro_auc: float
    int8_macro_auc: float
    auc_drop: float
    mean_prob_diff: float
    max_prob_diff: float
    float_logit_range: list
    output_logit_range: list
    per_class_auc_float: list
    per_class_auc_int8: list
    memory: dict
    passed: bool
    reasons: list


def verify(model: CNN1D, tflite_path: str | Path, x_test: np.ndarray, y_test: np.ndarray,
           cfg: Config, out_dir: str | Path | None = None) -> VerifyReport:
    float_logits = predict_logits(model, x_test, torch.device("cpu"))
    int8_logits, info = run_tflite(tflite_path, x_test)
    pf, pq = sigmoid(float_logits), sigmoid(int8_logits)

    auc_f, auc_q = macro_auc(y_test, pf), macro_auc(y_test, pq)
    drop = auc_f - auc_q
    mean_diff, max_diff = float(np.abs(pf - pq).mean()), float(np.abs(pf - pq).max())

    reasons = []
    if not np.isnan(drop) and drop > cfg.export.tolerance_auc:
        reasons.append(f"INT8 macro AUC dropped by {drop:.4f} (limit {cfg.export.tolerance_auc})")
    if mean_diff > cfg.export.tolerance_prob:
        reasons.append(f"mean probability difference {mean_diff:.4f} exceeds {cfg.export.tolerance_prob}")
    lo, hi = info["output_logit_range"]
    clipped = float(np.mean((float_logits < lo) | (float_logits > hi)))
    if clipped > 0.05:
        reasons.append(f"{clipped:.0%} of float logits fall outside the INT8 output range [{lo:.2f}, {hi:.2f}]")

    size = Path(tflite_path).stat().st_size
    report = VerifyReport(
        float_macro_auc=auc_f, int8_macro_auc=auc_q, auc_drop=drop,
        mean_prob_diff=mean_diff, max_prob_diff=max_diff,
        float_logit_range=[float(float_logits.min()), float(float_logits.max())],
        output_logit_range=[lo, hi],
        per_class_auc_float=per_class_auc(y_test, pf), per_class_auc_int8=per_class_auc(y_test, pq),
        memory=estimate_memory(size, model.in_channels, model.fc_size),
        passed=not reasons, reasons=reasons,
    )
    if out_dir:
        out_dir = Path(out_dir)
        (out_dir / "verify_report.json").write_text(json.dumps(asdict(report), indent=2))
        (out_dir / "verify_report.md").write_text(render_markdown(report, len(x_test)))
    return report


def render_markdown(r: VerifyReport, n_test: int) -> str:
    m = r.memory
    status = "PASSED" if r.passed else "FAILED"
    lines = [
        f"# ecg2mcu verification: {status}", "",
        f"Test windows: {n_test}", "",
        "| Metric | Value |", "|---|---|",
        f"| Float32 macro AUC | {r.float_macro_auc:.4f} |",
        f"| INT8 macro AUC | {r.int8_macro_auc:.4f} |",
        f"| AUC drop | {r.auc_drop:+.4f} |",
        f"| Mean / max probability difference | {r.mean_prob_diff:.4f} / {r.max_prob_diff:.4f} |",
        f"| Float logit range | [{r.float_logit_range[0]:.1f}, {r.float_logit_range[1]:.1f}] |",
        f"| INT8 output range | [{r.output_logit_range[0]:.1f}, {r.output_logit_range[1]:.1f}] |",
        f"| Flash (model) | {m['flash_kb']} KB |",
        f"| Tensor arena (lower bound / recommended) | {m['arena_lower_bound_kb']} KB / {m['arena_recommended_kb']} KB |",
        "",
    ]
    if r.reasons:
        lines += ["## Problems", *[f"- {reason}" for reason in r.reasons], ""]
    lines.append("The arena figures are estimates; confirm on the device (`AllocateTensors()` must succeed).")
    return "\n".join(lines) + "\n"
