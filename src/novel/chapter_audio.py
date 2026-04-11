from __future__ import annotations

import os
import re
from uuid import uuid4

from moviepy import AudioFileClip

from audio_merge import merge_wav_files
from classes.Tts import TTS
from config import ROOT_DIR


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
    merge_wav_files(segment_paths, merged_path)
    return segment_paths, durations, merged_path
