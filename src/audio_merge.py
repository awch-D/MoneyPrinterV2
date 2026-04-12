"""Concatenate WAV segments into one file (same sample rate, mono merge if needed)."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf


def _crossfade_concatenate(chunks: list[np.ndarray], sr: int, fade_ms: float) -> np.ndarray:
    """Linear crossfade at each junction; total length = sum(len) - (k-1)*n samples."""
    if fade_ms <= 0 or len(chunks) == 1:
        return np.concatenate(chunks)
    fade_samples = int(sr * fade_ms / 1000.0)
    fade_samples = max(0, min(fade_samples, int(sr * 0.08)))  # cap 80 ms
    out = np.asarray(chunks[0], dtype=np.float32).copy()
    for nxt in chunks[1:]:
        nxt = np.asarray(nxt, dtype=np.float32)
        n = min(
            fade_samples,
            max(0, len(out) // 2 - 1),
            max(0, len(nxt) // 2 - 1),
        )
        if n < 8:
            out = np.concatenate([out, nxt])
            continue
        w = np.linspace(0.0, 1.0, n, dtype=np.float32)
        blended = out[-n:] * (1.0 - w) + nxt[:n] * w
        out = np.concatenate([out[:-n], blended, nxt[n:]])
    return out


def merge_wav_files(paths: list[str], output_path: str, *, crossfade_ms: float = 0.0) -> str:
    """
    Concatenate WAV files in order. Resamples are not performed; all inputs must share sample_rate.

    When ``crossfade_ms`` > 0, applies a short linear crossfade at each boundary to reduce
    segment-edge clicks; output is slightly shorter than a naive concat — video pipelines
    should align segment durations to the merged file (e.g. ``_align_timeline_durations_to_merged_wav``).
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

    if crossfade_ms > 0 and len(arrays) > 1:
        merged = _crossfade_concatenate(arrays, sr or 24000, crossfade_ms)
    else:
        merged = np.concatenate(arrays)
    sf.write(output_path, merged, sr or 24000)
    return output_path
