"""Load .wav (mono real / stereo I-Q) and raw .iq/.bin/.raw/.dat captures."""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import hilbert

MAX_SAMPLES = 1_500_000

IQ_FORMATS = {
    "cf32": (np.dtype("f4"), 2, None),
    "ci16": (np.dtype("i2"), 2, 32768.0),
    "ci8": (np.dtype("i1"), 2, 128.0),
    "cu8": (np.dtype("u1"), 2, None),
}


def _validate_sample_rate(fs: float) -> float:
    try:
        rate = float(fs)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sample rate must be a finite number greater than zero.") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Sample rate must be a finite number greater than zero.")
    return rate


def load_wav(path: str):
    import soundfile as sf

    data, fs = sf.read(path, always_2d=True, dtype="float64")
    fs = _validate_sample_rate(fs)
    if data.shape[0] == 0:
        raise ValueError("WAV file contains no audio samples.")
    if not np.all(np.isfinite(data)):
        raise ValueError("WAV file contains non-finite audio samples.")
    if data.shape[1] >= 2:
        x = data[:, 0] + 1j * data[:, 1]
    else:
        x = hilbert(data[:, 0])  # real audio -> analytic signal
    return x[:MAX_SAMPLES].astype(np.complex128), fs


def load_iq(path: str, fs: float, dtype: str = "cf32", byte_order: str = "little"):
    fs = _validate_sample_rate(fs)
    dtype = dtype.lower()
    if dtype not in IQ_FORMATS:
        raise ValueError(f"Unknown IQ format '{dtype}'. Use one of {list(IQ_FORMATS)}")
    if byte_order not in ("little", "big"):
        raise ValueError("Byte order must be 'little' or 'big'.")
    sample_dtype, components_per_sample, scale = IQ_FORMATS[dtype]
    sample_bytes = sample_dtype.itemsize
    complex_sample_bytes = sample_bytes * components_per_sample
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValueError("IQ file is empty.")
    if file_size % complex_sample_bytes:
        raise ValueError(
            f"IQ file is truncated: its {file_size}-byte size does not contain complete "
            f"{dtype} complex samples ({complex_sample_bytes} bytes each)."
        )
    order = "<" if byte_order == "little" else ">"
    raw_dtype = sample_dtype.newbyteorder(order)
    raw = np.fromfile(path, dtype=raw_dtype, count=MAX_SAMPLES * components_per_sample)
    raw = raw.astype(np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("IQ file contains non-finite sample values.")
    if dtype == "cu8":
        raw = (raw - 127.5) / 127.5
    elif scale:
        raw = raw / scale
    return (raw[0::2] + 1j * raw[1::2]), float(fs)


def load_signal(
    path: str,
    iq_sample_rate: float = 2_000_000.0,
    iq_dtype: str = "cf32",
    byte_order: str = "little",
):
    """Returns (complex samples, sample rate, 'wav' | 'iq')."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        x, fs = load_wav(path)
        return x, fs, "wav"
    x, fs = load_iq(path, iq_sample_rate, iq_dtype, byte_order)
    return x, fs, "iq"
