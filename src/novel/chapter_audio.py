from __future__ import annotations

import os
import re
from uuid import uuid4

from moviepy import AudioFileClip

from audio_merge import merge_wav_files
from classes.Tts import TTS
from config import ROOT_DIR, get_audio_merge_crossfade_ms


def clean_narration_for_tts(text: str) -> str:
    """Match ShortVideoPipeline.generate_script_to_speech cleaning."""
    return re.sub(r"[^\w\s.?!。！？，,、；;：:\u4e00-\u9fff]", "", text)


def synthesize_segments_to_merged_wav(
    narrations: list[str],
    tts: TTS,
    *,
    work_dir: str | None = None,
) -> tuple[list[str], list[float], str]:
    """
    For each narration line, synthesize WAV, record duration, merge into one file.

    Returns:
        segment_wav_paths, segment_durations_seconds, merged_wav_path
    """
    base = work_dir or os.path.join(ROOT_DIR, ".mp")
    os.makedirs(base, exist_ok=True)

    segment_paths: list[str] = []
    durations: list[float] = []

    for text in narrations:
        cleaned = clean_narration_for_tts(text).strip()
        if not cleaned:
            raise RuntimeError("Empty narration after cleaning; cannot synthesize segment")
        seg_path = os.path.join(base, f"novel_seg_{uuid4()}.wav")
        tts.synthesize(cleaned, seg_path)
        segment_paths.append(seg_path)
        clip = AudioFileClip(seg_path)
        try:
            durations.append(float(clip.duration))
        finally:
            clip.close()

    merged_path = os.path.join(base, f"novel_merged_{uuid4()}.wav")
    merge_wav_files(segment_paths, merged_path, crossfade_ms=get_audio_merge_crossfade_ms())
    return segment_paths, durations, merged_path


def synthesize_full_track_to_wav(
    subtitle_lines: list[str],
    tts: TTS,
    *,
    work_dir: str | None = None,
) -> str:
    """
    One TTS pass on ``"".join(subtitle_lines)`` (no delimiter). Each line must be non-empty.

    ``subtitle_lines`` must match the same cleaning used for per-segment mode (e.g. from
    ``clean_narration_for_tts`` per narration).
    """
    base = work_dir or os.path.join(ROOT_DIR, ".mp")
    os.makedirs(base, exist_ok=True)
    lines = [s.strip() for s in subtitle_lines]
    if any(not s for s in lines):
        raise RuntimeError("Empty segment in subtitle_lines; cannot synthesize full track")
    full_text = "".join(lines)
    out_path = os.path.join(base, f"novel_full_{uuid4()}.wav")
    # One Gradio /do_job (or /do_job_t) call for the whole chapter — no client-side sentence splitting.
    tts.synthesize(full_text, out_path, qwen3_no_sentence_split=True)
    return out_path
