"""Use WhisperX forced alignment on one full narration WAV to derive per-segment durations.

Pipeline: single TTS pass on concatenated cleaned text → one WAV → WhisperX ``align`` with one
segment spanning the whole file → word/char timestamps → map back to each ``segment_texts`` row.

Requires optional dependency ``whisperx`` (and PyTorch). See ``docs/NovelChapter.md``.
"""

from __future__ import annotations

from typing import Any


def fallback_proportional_segment_durations(segment_texts: list[str], total_seconds: float) -> list[float]:
    """Split ``total_seconds`` by segment character length (for tests / emergency fallback)."""
    return _fallback_proportional_durations(segment_texts, total_seconds)


def _fallback_proportional_durations(segment_texts: list[str], total_seconds: float) -> list[float]:
    full_len = sum(max(1, len(s)) for s in segment_texts)
    if full_len <= 0:
        return [max(0.04, total_seconds)] if segment_texts else []
    return [max(0.04, total_seconds * max(1, len(s)) / full_len) for s in segment_texts]


def _durations_from_chars(chars: list[dict], segment_texts: list[str], full_ref: str) -> list[float] | None:
    if len(chars) != len(full_ref):
        return None
    out: list[float] = []
    off = 0
    for st in segment_texts:
        n = len(st)
        if n == 0:
            out.append(0.04)
            continue
        chunk = chars[off : off + n]
        off += n
        if not chunk:
            out.append(0.04)
            continue
        t0 = float(chunk[0].get("start", 0.0))
        t1 = float(chunk[-1].get("end", t0))
        out.append(max(0.04, t1 - t0))
    return out if off == len(full_ref) else None


def _durations_from_words(words: list[dict], segment_texts: list[str], full_ref: str) -> list[float]:
    """Assign word time ranges to character spans of full_ref (no spaces in ref)."""
    if not words:
        return _fallback_proportional_durations(segment_texts, 0.0)

    flat: list[tuple[str, float, float]] = []
    for w in words:
        txt = (w.get("word") or "").strip()
        if not txt:
            continue
        flat.append((txt, float(w.get("start", 0.0)), float(w.get("end", 0.0))))

    merged_chars: list[tuple[float, float]] = []
    for txt, ws, we in flat:
        span = max(we - ws, 1e-6)
        per = span / max(1, len(txt))
        for i, _ch in enumerate(txt):
            merged_chars.append((ws + i * per, ws + (i + 1) * per))

    if len(merged_chars) != len(full_ref):
        total = merged_chars[-1][1] if merged_chars else 0.0
        return _fallback_proportional_durations(segment_texts, total)

    out: list[float] = []
    off = 0
    for st in segment_texts:
        n = len(st)
        if n == 0:
            out.append(0.04)
            continue
        chunk = merged_chars[off : off + n]
        off += n
        t0, t1 = chunk[0][0], chunk[-1][1]
        out.append(max(0.04, t1 - t0))
    return out


def segment_durations_via_whisperx_align(
    wav_path: str,
    segment_texts: list[str],
    *,
    device: str = "cpu",
    language_code: str = "zh",
) -> list[float]:
    """
    Align ``"".join(segment_texts)`` to ``wav_path`` and return one duration per segment (seconds).

    Raises ``ImportError`` if whisperx is not installed; ``RuntimeError`` on alignment failure.
    """
    try:
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "novel_audio_pipeline=full_track_whisperx requires whisperx. "
            "Install with: pip install whisperx"
        ) from exc

    full_ref = "".join(segment_texts)
    if not full_ref.strip():
        raise RuntimeError("whisperx_align: empty transcript")

    audio = whisperx.load_audio(wav_path)
    try:
        from whisperx.audio import SAMPLE_RATE
    except Exception:
        SAMPLE_RATE = 16000
    duration = float(len(audio)) / float(SAMPLE_RATE)
    if duration <= 0.05:
        raise RuntimeError("whisperx_align: audio too short")

    segments_in: list[dict[str, Any]] = [{"text": full_ref, "start": 0.0, "end": duration}]
    align_model, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
    )
    result = whisperx.align(
        segments_in,
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=True,
    )

    segs = result.get("segments") or []
    if not segs:
        raise RuntimeError("whisperx_align: align returned no segments")

    seg0 = segs[0]
    chars = seg0.get("chars") or []
    if isinstance(chars, list) and chars:
        by_char = _durations_from_chars(chars, segment_texts, full_ref)
        if by_char is not None:
            return by_char

    words: list[dict] = []
    for s in segs:
        words.extend(s.get("words") or [])
    if not words:
        return _fallback_proportional_durations(segment_texts, duration)

    return _durations_from_words(words, segment_texts, full_ref)
