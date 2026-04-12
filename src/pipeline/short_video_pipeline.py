import glob
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import assemblyai as aai
from PIL import Image
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip

from classes.Tts import TTS
from config import (
    ROOT_DIR,
    equalize_subtitles,
    get_assemblyai_api_key,
    get_subtitle_bottom_margin,
    get_subtitle_caption_height_ratio,
    get_subtitle_font_color,
    get_subtitle_font_path,
    get_subtitle_font_size_config,
    get_subtitle_stroke_color,
    get_subtitle_stroke_width,
    get_imagemagick_path,
    get_script_sentence_length,
    get_stt_provider,
    get_threads,
    get_verbose,
    get_video_fps,
    get_video_ken_burns_enabled,
    get_video_ken_burns_pan_extent,
    get_video_ken_burns_pan_max_width_ratio,
    get_video_ken_burns_zoom_max,
    get_video_ken_burns_zoom_min,
    get_video_output_size,
    get_video_page_flip_duration_seconds,
    get_video_page_flip_probability,
    get_video_transition,
    get_video_transition_random_seed,
    get_whisper_cli_timeout_seconds,
    get_whisper_language,
    get_whisper_model_for_cli,
    resolve_whisper_cli_executable,
    whisper_cli_device_args,
    whisper_cli_uses_fp16,
)
from novel.image_style_presets import append_global_style_to_image_prompt
from providers.image_api_provider import ImageApiProvider
from providers.script_api_provider import ScriptApiProvider
from status import info, success, warning
from utils import choose_random_song
from video_motion import build_visual_timeline_clips

# MoviePy v2 removed change_settings. Keep compatibility by exporting a known binary path.
os.environ["IMAGEMAGICK_BINARY"] = get_imagemagick_path()


@dataclass
class VideoBuildResult:
    topic: str
    script: str
    video_path: str
    audio_path: str
    subtitle_path: str | None
    image_paths: list[str]


class ShortVideoPipeline:
    def __init__(self, script_provider: ScriptApiProvider, image_provider: ImageApiProvider):
        self.script_provider = script_provider
        self.image_provider = image_provider
        self.images: list[str] = []

    @staticmethod
    def _guess_image_extension(image_bytes: bytes) -> str:
        """Gemini image APIs often return JPEG; match gemini-image-gen skill magic-byte detection."""
        if len(image_bytes) >= 3 and image_bytes[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if len(image_bytes) >= 8 and image_bytes[:4] == b"\x89PNG":
            return ".png"
        if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return ".webp"
        return ".png"

    def _persist_image(self, image_bytes: bytes) -> str:
        ext = self._guess_image_extension(image_bytes)
        image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}{ext}")
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)
        self.images.append(image_path)
        return image_path

    def _placeholder_image_paths(self, count: int = 1) -> list[str]:
        """Solid slate when image API is skipped (size matches video_output_* / aspect)."""
        w, h = get_video_output_size()
        base = Image.new("RGB", (w, h), (14, 16, 22))
        paths: list[str] = []
        for _ in range(max(1, count)):
            path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}_placeholder.png")
            base.save(path, format="PNG")
            paths.append(path)
        return paths

    @staticmethod
    def _load_script_file(path: str) -> str:
        abs_path = path if os.path.isabs(path) else os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Script file not found: {abs_path}")
        with open(abs_path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise RuntimeError(f"Script file is empty: {abs_path}")
        return re.sub(r"\n{3,}", "\n\n", text)

    def generate_topic(self, niche: str, language: str) -> str:
        prompt = (
            "Generate one specific short-video topic sentence for this niche.\n"
            f"Niche: {niche}\n"
            f"Language: {language}\n"
            "Return only the topic sentence."
        )
        topic = self.script_provider.generate_text(prompt)
        return topic.strip()

    def generate_script(self, topic: str, language: str) -> str:
        sentence_length = get_script_sentence_length()
        prompt = f"""
        Generate a short-video narration script in exactly {sentence_length} short sentences.
        Subject: {topic}
        Language: {language}

        Rules:
        - No title
        - No markdown
        - No bullet points
        - No labels like narrator or voiceover
        - Return only raw script text
        """
        completion = self.script_provider.generate_text(prompt)
        completion = re.sub(r"\*", "", completion).strip()
        if not completion:
            raise RuntimeError("Script API returned empty script")
        return completion

    def generate_prompts(self, topic: str, script: str) -> list[str]:
        n_prompts = max(4, min(12, len(script) // 40))
        prompt = f"""
        Create {n_prompts} cinematic image prompts for a vertical short video.
        Topic: {topic}
        Script: {script}

        Return only a JSON array of strings.
        """
        completion = self.script_provider.generate_text(prompt).replace("```json", "").replace("```", "")
        try:
            data = json.loads(completion)
            if isinstance(data, dict) and "image_prompts" in data:
                prompts = data["image_prompts"]
            else:
                prompts = data
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Prompt generation returned invalid JSON: {completion[:300]}") from exc

        if not isinstance(prompts, list) or not prompts:
            raise RuntimeError("Prompt generation did not return a non-empty list")
        return [str(item).strip() for item in prompts if str(item).strip()]

    def generate_images(self, prompts: list[str]) -> list[str]:
        image_paths: list[str] = []
        total = len(prompts)
        for idx, prompt in enumerate(prompts, start=1):
            info(f"生图 {idx}/{total}：请求中…")
            sys.stdout.flush()
            image_bytes = self.image_provider.generate_image_bytes(
                append_global_style_to_image_prompt(prompt)
            )
            path = self._persist_image(image_bytes)
            image_paths.append(path)
            info(f"生图 {idx}/{total} 已完成 → {os.path.basename(path)}")
            sys.stdout.flush()
        if not image_paths:
            raise RuntimeError("No images were generated")
        return image_paths

    def generate_script_to_speech(self, script: str, tts: TTS) -> str:
        path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.wav")
        # 保留中文标点，只去掉特殊符号
        cleaned_script = re.sub(r"[^\w\s.?!。！？，,、；;：:\u4e00-\u9fff]", "", script)
        tts.synthesize(cleaned_script, path)
        return path

    def generate_subtitles_local_whisper(self, audio_path: str) -> str:
        """Run OpenAI ``whisper`` CLI (system install); not faster-whisper."""
        exe = resolve_whisper_cli_executable()
        if not exe:
            raise RuntimeError(
                "未找到本地 whisper 可执行文件。请安装 OpenAI Whisper（例如 "
                "`pip install -U openai-whisper` 或 `brew install openai-whisper`），"
                "或在 config.json 中设置 whisper_cli_path（如 /opt/homebrew/bin/whisper）。"
            )
        out_dir = os.path.join(ROOT_DIR, ".mp", f"whisper_cli_{uuid4()}")
        os.makedirs(out_dir, exist_ok=True)
        audio_abs = os.path.abspath(audio_path)
        if not os.path.isfile(audio_abs):
            raise FileNotFoundError(f"Whisper 输入音频不存在: {audio_abs}")

        cmd: list[str] = [
            exe,
            audio_abs,
            "--model",
            get_whisper_model_for_cli(),
            "--output_dir",
            out_dir,
            "--output_format",
            "srt",
            "--fp16",
            "True" if whisper_cli_uses_fp16() else "False",
        ]
        cmd.extend(whisper_cli_device_args())
        thr = get_threads()
        if thr and int(thr) > 0:
            cmd.extend(["--threads", str(max(1, int(thr)))])
        lang = get_whisper_language()
        if lang:
            cmd.extend(["--language", lang])
        if not get_verbose():
            cmd.extend(["--verbose", "False"])

        info(f"Whisper CLI 转写中：{exe}（输出目录 {out_dir}）")
        sys.stdout.flush()
        timeout_s = float(get_whisper_cli_timeout_seconds())
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Whisper CLI 超时（{int(timeout_s)} 秒）。可在 config.json 增大 whisper_cli_timeout_seconds。"
            ) from exc

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Whisper CLI 失败（exit {proc.returncode}）：{err[:4000]}")

        base = os.path.splitext(os.path.basename(audio_abs))[0]
        srt_candidate = os.path.join(out_dir, f"{base}.srt")
        if not os.path.isfile(srt_candidate):
            found = sorted(glob.glob(os.path.join(out_dir, "*.srt")))
            if not found:
                raise RuntimeError(f"Whisper 未生成 .srt（目录：{out_dir}）")
            srt_candidate = found[0]

        srt_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.srt")
        shutil.copy2(srt_candidate, srt_path)
        shutil.rmtree(out_dir, ignore_errors=True)
        return srt_path

    def generate_subtitles_assemblyai(self, audio_path: str) -> str:
        api_key = get_assemblyai_api_key()
        if not api_key:
            raise RuntimeError("stt_provider=third_party_assemblyai but assembly_ai_api_key is missing")
        aai.settings.api_key = api_key
        transcript = aai.Transcriber(config=aai.TranscriptionConfig()).transcribe(audio_path)
        subtitles = transcript.export_subtitles_srt()
        srt_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.srt")
        with open(srt_path, "w", encoding="utf-8") as file:
            file.write(subtitles)
        return srt_path

    def generate_subtitles(self, audio_path: str) -> str:
        provider = str(get_stt_provider() or "local_whisper").lower()
        if provider == "third_party_assemblyai":
            return self.generate_subtitles_assemblyai(audio_path)
        return self.generate_subtitles_local_whisper(audio_path)

    def _subtitle_textclip_factory(self, out_w: int, out_h: int):
        min_dim = min(out_w, out_h)
        cfg_sub_size = get_subtitle_font_size_config()
        subtitle_font_size = (
            cfg_sub_size
            if cfg_sub_size is not None
            else max(34, int(62 * min_dim / 1080))
        )
        subtitle_font = get_subtitle_font_path()
        cap_ratio = get_subtitle_caption_height_ratio()
        caption_h = max(96, int(out_h * cap_ratio))
        bottom_margin = get_subtitle_bottom_margin()
        subtitle_y = max(0, out_h - caption_h - bottom_margin)
        stroke_w = get_subtitle_stroke_width()
        stroke_col = get_subtitle_stroke_color()
        if stroke_col is None:
            stroke_w = 0

        def make_clip(txt: str) -> TextClip:
            return TextClip(
                text=txt,
                font=subtitle_font,
                font_size=subtitle_font_size,
                color=get_subtitle_font_color(),
                stroke_color=stroke_col,
                stroke_width=stroke_w,
                size=(out_w, caption_h),
                method="caption",
                horizontal_align="center",
                vertical_align="bottom",
            )

        return make_clip, subtitle_y

    @staticmethod
    def _align_timeline_durations_to_merged_wav(
        segment_durations: list[float],
        narration_wav_path: str,
        eps: float = 0.08,
    ) -> list[float]:
        """Scale segment lengths so visual timeline matches merged narration WAV duration."""
        total = sum(segment_durations)
        if total <= 0:
            return segment_durations
        ac = AudioFileClip(narration_wav_path)
        try:
            audio_dur = float(ac.duration or 0.0)
        finally:
            ac.close()
        if audio_dur <= 0:
            return segment_durations
        if abs(total - audio_dur) <= eps:
            return segment_durations
        scale = audio_dur / total
        return [max(0.04, float(d) * scale) for d in segment_durations]

    @staticmethod
    def _fit_image_clip(image_path: str, duration: float, out_w: int, out_h: int, fps: int = 30) -> Any:
        target_ratio = out_w / out_h
        clip = ImageClip(image_path).with_duration(duration).with_fps(fps)
        r = clip.w / clip.h if clip.h else target_ratio
        if r < target_ratio:
            clip = clip.cropped(
                width=clip.w,
                height=max(1, round(clip.w / target_ratio)),
                x_center=clip.w / 2,
                y_center=clip.h / 2,
            )
        else:
            clip = clip.cropped(
                width=max(1, round(clip.h * target_ratio)),
                height=clip.h,
                x_center=clip.w / 2,
                y_center=clip.h / 2,
            )
        return clip.resized((out_w, out_h))

    def _compose_still_sequence(
        self,
        image_paths: list[str],
        segment_durations: list[float],
        out_w: int,
        out_h: int,
    ) -> Any:
        """Ken Burns + optional page-flip transitions; durations must match narration timeline."""
        fps = get_video_fps()
        return build_visual_timeline_clips(
            image_paths,
            [float(d) for d in segment_durations],
            out_w,
            out_h,
            fps=fps,
            ken_burns=get_video_ken_burns_enabled(),
            zoom_min=get_video_ken_burns_zoom_min(),
            zoom_max=get_video_ken_burns_zoom_max(),
            pan_extent=get_video_ken_burns_pan_extent(),
            pan_max_width_ratio=get_video_ken_burns_pan_max_width_ratio(),
            transition_mode=get_video_transition(),
            page_flip_probability=get_video_page_flip_probability(),
            page_flip_duration_seconds=get_video_page_flip_duration_seconds(),
            transition_random_seed=get_video_transition_random_seed(),
        )

    def _finalize_with_subtitles_and_bgm(
        self,
        video_clip: Any,
        tts_path: str,
        output_path: str,
    ) -> str | None:
        """Attach narration + optional BGM, burn subtitles, write MP4. Closes tts_clip after write."""
        threads = get_threads()
        out_w, out_h = get_video_output_size()
        tts_clip = AudioFileClip(tts_path)
        narration_duration = float(tts_clip.duration)

        make_textclip, subtitle_y = self._subtitle_textclip_factory(out_w, out_h)

        subtitle_path: str | None = None
        subtitles = None
        try:
            subtitle_path = self.generate_subtitles(tts_path)
            equalize_subtitles(subtitle_path, 10)
            subtitles = (
                SubtitlesClip(subtitle_path, make_textclip=make_textclip)
                .with_position(("center", subtitle_y))
                .with_duration(narration_duration)
            )
        except Exception as exc:
            warning(f"Subtitles unavailable, continuing without them: {exc}")

        bgm_path = choose_random_song()
        narration_audio = tts_clip.with_fps(44100)
        audio_tracks = [narration_audio]
        if bgm_path:
            bgm = AudioFileClip(bgm_path).with_fps(44100).with_effects([afx.MultiplyVolume(0.1)])
            bgm_len = float(bgm.duration or 0.0)
            if bgm_len > narration_duration + 1e-3:
                bgm = bgm.subclipped(0, narration_duration)
            audio_tracks.append(bgm)
        # 成片时长只跟对白走：BGM 过长已截断；合成轨再钳到对白时长，避免取最长子轨
        final_audio = CompositeAudioClip(audio_tracks).with_duration(narration_duration)
        base = video_clip.with_duration(narration_duration).with_audio(final_audio)

        if subtitles is not None:
            final_clip = CompositeVideoClip([base, subtitles])
        else:
            final_clip = base

        # MP4: H.264 + AAC + yuv420p + faststart = standard single-file deliverable (not MP3-in-MP4).
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".mp4":
            # libx264 without CRF/bitrate can yield very low average bitrate on some runs
            # (tiny MP4 + blocky video). CRF gives stable visual quality across machines.
            final_clip.write_videofile(
                output_path,
                threads=threads,
                codec="libx264",
                audio_codec="aac",
                audio_bitrate="192k",
                preset="medium",
                pixel_format="yuv420p",
                ffmpeg_params=["-crf", "20", "-movflags", "+faststart"],
            )
        else:
            final_clip.write_videofile(output_path, threads=threads)
        success(f'Wrote video to "{output_path}"')
        return subtitle_path

    def combine(self, image_paths: list[str], tts_path: str) -> tuple[str, str | None]:
        output_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.mp4")
        tts_clip = AudioFileClip(tts_path)
        max_duration = tts_clip.duration
        tts_clip.close()
        req_dur = max_duration / len(image_paths)
        out_w, out_h = get_video_output_size()

        if get_verbose():
            info("[+] Combining image clips...")

        paths_order: list[str] = []
        durs_order: list[float] = []
        total_duration = 0.0
        while total_duration < max_duration:
            for image_path in image_paths:
                paths_order.append(image_path)
                durs_order.append(req_dur)
                total_duration += req_dur
                if total_duration >= max_duration:
                    break

        final_clip = self._compose_still_sequence(paths_order, durs_order, out_w, out_h)
        subtitle_path = self._finalize_with_subtitles_and_bgm(final_clip, tts_path, output_path)
        return output_path, subtitle_path

    def combine_timeline(
        self,
        image_paths: list[str],
        segment_durations: list[float],
        narration_wav_path: str,
    ) -> tuple[str, str | None]:
        """
        One still per segment; each clip lasts exactly the matching segment duration.
        Narration WAV must span the same total time (concatenated segment TTS).
        """
        if len(image_paths) != len(segment_durations):
            raise ValueError(
                f"combine_timeline: len(image_paths)={len(image_paths)} != len(segment_durations)={len(segment_durations)}"
            )
        if not image_paths:
            raise ValueError("combine_timeline: empty timeline")
        output_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.mp4")
        out_w, out_h = get_video_output_size()

        raw_durs = [float(d) for d in segment_durations]
        aligned_durations = self._align_timeline_durations_to_merged_wav(
            raw_durs,
            narration_wav_path,
        )
        if get_verbose() and abs(sum(raw_durs) - sum(aligned_durations)) > 0.05:
            warning(
                "Scaled per-segment clip durations to match merged narration WAV "
                f"({sum(raw_durs):.2f}s → {sum(aligned_durations):.2f}s)."
            )

        if get_verbose():
            info("[+] Combining image clips (per-segment timeline)...")

        final_clip = self._compose_still_sequence(image_paths, aligned_durations, out_w, out_h)
        subtitle_path = self._finalize_with_subtitles_and_bgm(final_clip, narration_wav_path, output_path)
        return output_path, subtitle_path

    def run(
        self,
        niche: str,
        language: str,
        topic: str | None = None,
        *,
        script_file: str | None = None,
    ) -> VideoBuildResult:
        if script_file:
            if get_verbose():
                info("Pipeline (text file): script file → TTS → subtitles + BGM → video (placeholder visuals).")
            script = self._load_script_file(script_file)
            resolved_topic = (topic or "").strip() or os.path.basename(script_file)
            if get_verbose():
                info(f'Topic label: "{resolved_topic}"')
                info("Skipping script / image APIs; using solid placeholder frame(s).")
            image_paths = self._placeholder_image_paths(1)
        else:
            if get_verbose():
                info("Pipeline: script → image prompts → images → TTS → subtitles → video.")
            resolved_topic = topic or self.generate_topic(niche, language)
            if get_verbose():
                info(f'Topic: "{resolved_topic}"')
                info("Calling script API to generate narration (this can take a while)...")
            script = self.generate_script(resolved_topic, language)
            if get_verbose():
                info("Generating image prompts via script API...")
            prompts = self.generate_prompts(resolved_topic, script)
            if get_verbose():
                info(f"Generating {len(prompts)} images...")
            image_paths = self.generate_images(prompts)

        if get_verbose():
            info("Synthesizing speech (TTS)...")
        audio_path = self.generate_script_to_speech(script, TTS())
        if get_verbose():
            info("Composing final video (encoding may take several minutes)...")
        video_path, subtitle_path = self.combine(image_paths, audio_path)
        return VideoBuildResult(
            topic=resolved_topic,
            script=script,
            video_path=os.path.abspath(video_path),
            audio_path=os.path.abspath(audio_path),
            subtitle_path=os.path.abspath(subtitle_path) if subtitle_path else None,
            image_paths=[os.path.abspath(path) for path in image_paths],
        )
