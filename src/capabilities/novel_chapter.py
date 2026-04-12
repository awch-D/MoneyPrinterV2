from __future__ import annotations

import os
import sys

from classes.Tts import TTS
from capabilities.base import RunContext
from config import ROOT_DIR, get_tts_backend, get_verbose
from novel.chapter_analyzer import analyze_chapter, build_merged_image_prompt
from novel.chapter_audio import synthesize_segments_to_merged_wav
from pipeline.short_video_pipeline import ShortVideoPipeline, VideoBuildResult
from providers.image_api_provider import ImageApiProvider
from providers.script_api_provider import ScriptApiProvider
from status import info, success


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
        if get_verbose():
            info("Synthesizing per-segment TTS and merging audio...")
        tts_engine = TTS()
        if get_verbose():
            info(
                f"Novel chapter TTS: single engine instance (backend={get_tts_backend()!r}); "
                "every segment uses the same configured voice / reference."
            )
        _seg_wavs, durations, merged_wav = synthesize_segments_to_merged_wav(narrations, tts_engine)

        if get_verbose():
            info("Composing timeline video (per-segment durations)...")
        video_path, subtitle_path = pipeline.combine_timeline(image_paths, durations, merged_wav)

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
