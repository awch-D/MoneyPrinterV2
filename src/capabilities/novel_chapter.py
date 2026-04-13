from __future__ import annotations

import json
import os
import sys

from classes.Tts import TTS
from capabilities.base import RunContext
from config import (
    ROOT_DIR,
    get_novel_audio_pipeline,
    get_qwen3_tts_reference_audio,
    get_tts_backend,
    get_verbose,
    get_whisperx_device,
    get_whisperx_language_code,
)
from novel.chapter_analyzer import analyze_chapter, build_merged_image_prompt
from novel.chapter_audio import (
    clean_narration_for_tts,
    synthesize_full_track_to_wav,
    synthesize_segments_to_merged_wav,
)
from novel.whisperx_segment_align import segment_durations_via_whisperx_align
from pipeline.short_video_pipeline import ShortVideoPipeline, VideoBuildResult
from providers.image_api_provider import ImageApiProvider
from providers.script_api_provider import ScriptApiProvider
from status import info, success, warning


class NovelChapterCapability:
    name = "novel_chapter"

    def run(self, ctx: RunContext) -> VideoBuildResult:
        chapter_path = (ctx.chapter_file or "").strip()
        if not chapter_path:
            raise ValueError("novel_chapter capability requires --chapter-file")

        script_provider = ScriptApiProvider()
        image_provider = ImageApiProvider()
        pipeline = ShortVideoPipeline(script_provider=script_provider, image_provider=image_provider)

        chapter_text = ShortVideoPipeline._load_script_file(chapter_path)
        topic_label = (ctx.topic or "").strip() or os.path.basename(chapter_path)

        if get_verbose():
            info(f'Novel chapter pipeline: "{topic_label}" ({ctx.language})')

        plan = analyze_chapter(chapter_text, ctx.language, script_provider)
        if get_verbose():
            info(f"Storyboard: {len(plan.characters)} characters, {len(plan.segments)} segments")

        image_paths: list[str] = []
        total_seg = len(plan.segments)
        for i, seg in enumerate(plan.segments, start=1):
            merged_prompt = build_merged_image_prompt(plan, seg)
            info(f"生图 {i}/{total_seg}：请求中…")
            sys.stdout.flush()
            image_bytes = image_provider.generate_image_bytes(merged_prompt)
            path = pipeline._persist_image(image_bytes)
            image_paths.append(path)
            info(f"生图 {i}/{total_seg} 已完成 → {os.path.basename(path)}")
            sys.stdout.flush()

        narrations = [s.narration for s in plan.segments]
        subtitle_lines = [clean_narration_for_tts(t).strip() for t in narrations]
        if any(not line for line in subtitle_lines):
            raise RuntimeError("One or more segments have empty narration after cleaning")

        if get_verbose():
            info("Synthesizing speech (novel chapter)…")
        tts_engine = TTS()
        if get_verbose():
            info(
                f"Novel chapter TTS: single engine instance (backend={get_tts_backend()!r}); "
                "every segment uses the same configured voice / reference."
            )
        backend = get_tts_backend().strip().lower()
        if backend in ("qwen3", "qwen3_http", "qwen3_gradio"):
            ref = get_qwen3_tts_reference_audio()
            if not ref or not os.path.isfile(ref):
                warning(
                    "qwen3_tts_reference_audio 未配置或不是有效文件时，分段 TTS 音色容易飘；"
                    "建议在 config.json 设置参考干声以稳定音色。"
                )

        audio_mode = get_novel_audio_pipeline()
        if audio_mode == "full_track_whisperx":
            if get_verbose():
                info(
                    "Novel audio: full_track_whisperx — single TTS pass + WhisperX forced alignment "
                    f"(device={get_whisperx_device()!r}, lang={get_whisperx_language_code()!r})"
                )
            merged_wav = synthesize_full_track_to_wav(subtitle_lines, tts_engine)
            _seg_wavs = [merged_wav]
            durations = segment_durations_via_whisperx_align(
                merged_wav,
                subtitle_lines,
                device=get_whisperx_device(),
                language_code=get_whisperx_language_code(),
            )
        else:
            if get_verbose():
                info("Novel audio: segment_merge — per-segment TTS + WAV merge")
            _seg_wavs, durations, merged_wav = synthesize_segments_to_merged_wav(narrations, tts_engine)

        manifest_path = os.path.join(ROOT_DIR, ".mp", "last_timeline_manifest.json")
        try:
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "topic": topic_label,
                        "chapter_file": os.path.abspath(chapter_path),
                        "merged_wav": os.path.abspath(merged_wav),
                        "segment_durations": [float(x) for x in durations],
                        "segment_wavs": [os.path.abspath(p) for p in _seg_wavs],
                        "image_paths": [os.path.abspath(p) for p in image_paths],
                        "subtitle_segment_texts": subtitle_lines,
                        "novel_audio_pipeline": audio_mode,
                    },
                    mf,
                    ensure_ascii=False,
                    indent=2,
                )
            if get_verbose():
                info(f"Timeline manifest for recombine: {manifest_path}")
        except OSError as exc:
            if get_verbose():
                warning(f"Could not write timeline manifest: {exc}")

        if get_verbose():
            info("Composing timeline video (per-segment durations)...")
        video_path, subtitle_path = pipeline.combine_timeline(
            image_paths,
            durations,
            merged_wav,
            subtitle_segment_texts=subtitle_lines,
        )

        full_script = plan.full_script()
        success(f'Novel chapter episode complete: "{topic_label}"')

        return VideoBuildResult(
            topic=topic_label,
            script=full_script,
            video_path=os.path.abspath(video_path),
            audio_path=os.path.abspath(merged_wav),
            subtitle_path=os.path.abspath(subtitle_path) if subtitle_path else None,
            image_paths=[os.path.abspath(p) for p in image_paths],
        )
