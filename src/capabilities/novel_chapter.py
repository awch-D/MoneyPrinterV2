from __future__ import annotations

import json
import os
import sys

from classes.Tts import TTS
from capabilities.base import RunContext
from config import (
    ROOT_DIR,
    get_novel_audio_pipeline,
    get_novel_tts_punctuate_enabled,
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

def _placeholder_images(pipeline: ShortVideoPipeline, count: int) -> list[str]:
    """Generate placeholder frames when image API is unavailable."""
    from io import BytesIO

    from PIL import Image, ImageDraw

    paths: list[str] = []
    w, h = 1920, 1080
    for i in range(1, count + 1):
        img = Image.new("RGB", (w, h), (245, 245, 245))
        d = ImageDraw.Draw(img)
        text = f"Placeholder image\\nSegment {i}/{count}"
        d.multiline_text((80, 80), text, fill=(20, 20, 20), spacing=10)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)
        path = pipeline._persist_image(buf.getvalue())
        paths.append(path)
    return paths


def _punctuate_for_qwen3(lines: list[str]) -> list[str]:
    """
    Heuristic punctuation enhancement for TTS prosody.
    Keep content same; add pauses/turns via punctuation that Qwen3 is sensitive to.
    """
    out: list[str] = []
    for s in lines:
        t = (s or "").strip()
        if not t:
            out.append(t)
            continue
        # Add a short pause for very short declarative lines.
        if len(t) <= 8 and t[-1] not in "。！？!?…":
            t = t + "。"
        # Add a dramatic pause for emphatic single-clause lines.
        if "，" not in t and "。" in t and len(t) <= 18 and not t.endswith("……"):
            t = t.replace("。", "……。")
        out.append(t)
    return out


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
        
        # 生成章节 hash 用于文件命名
        chapter_hash = os.path.basename(chapter_path).replace(".txt", "").replace(" ", "_")
        
        # 立即保存章节分析结果和完整的图片生成计划（manifest）
        analysis_path = os.path.join(ROOT_DIR, ".mp", "last_chapter_analysis.json")
        mp_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(mp_dir, exist_ok=True)
        
        # 预先生成所有图片的路径（固定序号）
        total_seg = len(plan.segments)
        planned_image_paths = []
        for i in range(1, total_seg + 1):
            img_name = f"{chapter_hash}_seg_{i:03d}.jpg"
            img_path = os.path.join(mp_dir, img_name)
            planned_image_paths.append(img_path)
        
        try:
            with open(analysis_path, "w", encoding="utf-8") as af:
                json.dump(
                    {
                        "topic": topic_label,
                        "chapter_file": os.path.abspath(chapter_path),
                        "chapter_hash": chapter_hash,
                        "language": ctx.language,
                        "style_bible": plan.style_bible,
                        "total_segments": total_seg,
                        "planned_image_paths": planned_image_paths,
                        "characters": [{"id": c.id, "name": c.name, "look": c.look} for c in plan.characters],
                        "segments": [
                            {
                                "narration": s.narration,
                                "scene_summary": s.scene_summary,
                                "image_prompt": s.image_prompt,
                                "visible_character_ids": s.visible_character_ids,
                            }
                            for s in plan.segments
                        ],
                    },
                    af,
                    ensure_ascii=False,
                    indent=2,
                )
            if get_verbose():
                info(f"Chapter analysis & manifest saved: {analysis_path}")
        except OSError as exc:
            if get_verbose():
                warning(f"Could not save chapter analysis: {exc}")

        image_paths: list[str] = []
        if ctx.placeholder_images:
            image_paths = _placeholder_images(pipeline, len(plan.segments))
            if get_verbose():
                info(f"Skipping image generation; using {len(image_paths)} placeholder images.")
        elif ctx.reuse_images_manifest:
            manifest_path = os.path.join(ROOT_DIR, ".mp", "last_timeline_manifest.json")
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError(
                    "reuse-images-manifest is set but .mp/last_timeline_manifest.json not found. "
                    "Run once with images, or disable --reuse-images-manifest."
                )
            with open(manifest_path, encoding="utf-8") as f:
                m = json.load(f)
            manifest_chapter = str(m.get("chapter_file", "")).strip()
            expected = os.path.abspath(chapter_path if os.path.isabs(chapter_path) else os.path.abspath(chapter_path))
            if manifest_chapter and os.path.abspath(manifest_chapter) != expected:
                warning(
                    "Reusing images from manifest for a different chapter_file. "
                    f"manifest={manifest_chapter!r}, current={expected!r}"
                )
            manifest_images = m.get("image_paths", [])
            if not isinstance(manifest_images, list) or not all(isinstance(p, str) for p in manifest_images):
                raise RuntimeError("Timeline manifest image_paths is missing or invalid")
            image_paths = [os.path.abspath(p) for p in manifest_images]
            missing = [p for p in image_paths if not os.path.isfile(p)]
            if missing:
                warning(
                    "Manifest image_paths are missing on disk; generating placeholder images to proceed. "
                    "Use --keep-temp after a successful image run to preserve frames for reuse."
                )
                image_paths = _placeholder_images(pipeline, len(plan.segments))
            if len(image_paths) != len(plan.segments):
                seg_n = len(plan.segments)
                if len(image_paths) > seg_n:
                    warning(
                        f"Manifest image count ({len(image_paths)}) > storyboard segments ({seg_n}); "
                        "truncating to match segments."
                    )
                    image_paths = image_paths[:seg_n]
                else:
                    warning(
                        f"Manifest image count ({len(image_paths)}) < storyboard segments ({seg_n}); "
                        "padding with placeholder images."
                    )
                    image_paths = image_paths + _placeholder_images(
                        pipeline, seg_n - len(image_paths)
                    )
            if get_verbose():
                info(f"Skipping image generation; reusing {len(image_paths)} images from {manifest_path}")
        else:
            # 使用 manifest 中的固定路径生成图片，支持断点续传
            image_paths = planned_image_paths
            
            # 检查已存在的图片
            existing_count = 0
            for img_path in image_paths:
                if os.path.isfile(img_path):
                    existing_count += 1
                else:
                    break
            
            if existing_count > 0:
                info(f"发现 {existing_count}/{total_seg} 张已生成的图片，从第 {existing_count + 1} 张继续")
            
            # 生成剩余图片
            for i in range(existing_count, total_seg):
                seg = plan.segments[i]
                img_path = image_paths[i]
                img_name = os.path.basename(img_path)
                
                merged_prompt = build_merged_image_prompt(plan, seg)
                info(f"生图 {i + 1}/{total_seg}：请求中…")
                sys.stdout.flush()
                
                try:
                    image_bytes = image_provider.generate_image_bytes(merged_prompt)
                    
                    # 直接保存到 manifest 中预定的路径
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    info(f"生图 {i + 1}/{total_seg} 已完成 → {img_name}")
                    sys.stdout.flush()
                except Exception as e:
                    warning(f"生图 {i + 1}/{total_seg} 失败: {e}")
                    info(f"已生成 {i}/{total_seg} 张图片，可以重新运行继续生成")
                    raise

        narrations = [s.narration for s in plan.segments]
        subtitle_lines = [clean_narration_for_tts(t).strip() for t in narrations]
        if any(not line for line in subtitle_lines):
            raise RuntimeError("One or more segments have empty narration after cleaning")

        # Optionally enhance punctuation for TTS prosody.
        # Use the same text for TTS + WhisperX alignment; keep displayed subtitles as original cleaned lines.
        tts_lines = subtitle_lines
        if get_novel_tts_punctuate_enabled():
            if get_verbose():
                info("Novel TTS: punctuation enhancement enabled (novel_tts_punctuate=true).")
            tts_lines = _punctuate_for_qwen3(subtitle_lines)

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
                    "若 qwen3_tts_api_name 为 /do_job，缺少有效参考文件会在合成前直接报错。"
                )

        audio_mode = get_novel_audio_pipeline()
        if audio_mode == "full_track_whisperx":
            if get_verbose():
                info(
                    "Novel audio: full_track_whisperx — single TTS pass + WhisperX forced alignment "
                    f"(device={get_whisperx_device()!r}, lang={get_whisperx_language_code()!r})"
                )
            merged_wav = synthesize_full_track_to_wav(tts_lines, tts_engine)
            _seg_wavs = [merged_wav]
            durations = segment_durations_via_whisperx_align(
                merged_wav,
                tts_lines,
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
