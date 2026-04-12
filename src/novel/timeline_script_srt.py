"""Build SRT from per-segment narration text and segment durations (TTS timeline).

Used when subtitles must match the script fed to TTS, without ASR on the merged WAV.
"""

from __future__ import annotations

import os
from uuid import uuid4


def seconds_to_srt_timestamp(total_seconds: float) -> str:
    """SRT timecode: HH:MM:SS,mmm (non-negative)."""
    t = max(0.0, float(total_seconds))
    hours = int(t // 3600)
    t -= hours * 3600
    minutes = int(t // 60)
    t -= minutes * 60
    whole = int(t)
    ms = int(round((t - whole) * 1000))
    if ms >= 1000:
        whole += 1
        ms = 0
    if whole >= 60:
        whole -= 60
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        hours += 1
    return f"{hours:02d}:{minutes:02d}:{whole:02d},{ms:03d}"


def write_timeline_script_subtitles_srt(
    segment_texts: list[str],
    segment_durations: list[float],
    srt_path: str | None = None,
) -> str:
    """
    One SRT cue per timeline segment; times follow cumulative segment_durations.

    Text should be the same strings passed to TTS (e.g. after clean_narration_for_tts).

    Args:
        segment_texts: One line per segment (cleaned narration).
        segment_durations: Same length as segment_texts; seconds per segment (e.g. aligned durations).
        srt_path: Output file; default ``<project>/.mp/<uuid>.srt`` (uses ``config.ROOT_DIR``).

    Returns:
        Absolute path to the written SRT file.
    """
    if len(segment_texts) != len(segment_durations):
        raise ValueError(
            "write_timeline_script_subtitles_srt: len(texts) != len(durations) "
            f"({len(segment_texts)} vs {len(segment_durations)})"
        )
    if srt_path is None:
        from config import ROOT_DIR

        srt_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.srt")
    srt_path = os.path.abspath(srt_path)
    os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)

    cursor = 0.0
    blocks: list[str] = []
    for i, (raw, dur) in enumerate(zip(segment_texts, segment_durations, strict=True), start=1):
        d = max(0.04, float(dur))
        start_t = cursor
        end_t = cursor + d
        cursor = end_t
        body = (raw or "").strip().replace("\r\n", "\n").replace("\r", "\n")
        if not body:
            body = "…"
        blocks.append(
            f"{i}\n"
            f"{seconds_to_srt_timestamp(start_t)} --> {seconds_to_srt_timestamp(end_t)}\n"
            f"{body}\n"
        )
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
        if blocks:
            f.write("\n")
    return srt_path
