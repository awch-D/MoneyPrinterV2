"""Concatenate WAV segments into one file (same sample rate, mono merge if needed)."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf


def merge_wav_files(paths: list[str], output_path: str) -> str:
    """
    Concatenate WAV files in order. Resamples are not performed; all inputs must share sample_rate.
    """
    if not paths:
        raise ValueError("merge_wav_files: empty paths")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    arrays: list[np.ndarray] = []
    sr: int | None = None
    for path in paths:
        data, file_sr = sf.read(path)
        if sr is None:
            sr = int(file_sr)
        elif int(file_sr) != sr:
            raise RuntimeError(
                f"Sample rate mismatch in merge_wav_files: {path} has {file_sr}, expected {sr}"
            )
        if data.ndim > 1:
            data = data.mean(axis=1)
        arrays.append(np.asarray(data, dtype=np.float32))

    merged = np.concatenate(arrays)
    sf.write(output_path, merged, sr or 24000)
    return output_path
