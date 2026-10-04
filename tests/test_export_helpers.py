import re

import numpy as np

from ecg2mcu.export import c_header, estimate_memory


def test_header_roundtrip():
    data = bytes(range(256)) * 3 + b"TFL3"
    text = c_header(data, "my_model")
    assert "alignas(8) const unsigned char my_model[] PROGMEM" in text
    assert f"const unsigned int my_model_len = {len(data)};" in text
    body = text[text.index("{") + 1:text.rindex("};")]
    parsed = bytes(int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})", body))
    assert parsed == data


def test_memory_estimate_matches_thesis_model():
    mem = estimate_memory(model_bytes=227 * 1024, in_channels=1, fc_size=256)
    assert mem["flash_kb"] == 227.0
    assert mem["arena_lower_bound_kb"] == 46.9       # 48,000 bytes: conv2, 500x32 in + 500x64 out
    assert 64 <= mem["arena_recommended_kb"] <= 80   # thesis measured 68 KB on the device


def test_memory_grows_with_channels_only_at_the_input():
    one = estimate_memory(1, 1, 256)["arena_lower_bound_kb"]
    twelve = estimate_memory(1, 12, 256)["arena_lower_bound_kb"]
    assert one == twelve  # the peak is inside the network, not at the input
