import json
import os
import re
from dataclasses import dataclass
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
    concatenate_videoclips,
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
    get_video_output_size,
    get_whisper_compute_type,
    get_whisper_device,
    get_whisper_language,
    get_whisper_model,
    get_whisper_model_path,
)
from providers.image_api_provider import ImageApiProvider
from providers.script_api_provider import ScriptApiProvider
from status import info, success, warning
from utils import choose_random_song

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

    def _persist_image(self, image_bytes: bytes) -> str:
        image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
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
        for prompt in prompts:
            image_bytes = self.image_provider.generate_image_bytes(prompt)
            image_paths.append(self._persist_image(image_bytes))
        if not image_paths:
            raise RuntimeError("No images were generated")
        return image_paths

    def generate_script_to_speech(self, script: str, tts: TTS) -> str:
        path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.wav")
        # 保留中文标点，只去掉特殊符号
        cleaned_script = re.sub(r"[^\w\s.?!。！？，,、；;：:\u4e00-\u9fff]", "", script)
        tts.synthesize(cleaned_script, path)
        return path

    def _format_srt_timestamp(self, seconds: float) -> str:
        total_millis = max(0, int(round(seconds * 1000)))
        hours = total_millis // 3600000
        minutes = (total_millis % 3600000) // 60000
        secs = (total_millis % 60000) // 1000
        millis = total_millis % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def generate_subtitles_local_whisper(self, audio_path: str) -> str:
        from faster_whisper import WhisperModel

        model_path = get_whisper_model_path()
        model_id = model_path if model_path else get_whisper_model()
        model = WhisperModel(
            model_id,
            device=get_whisper_device(),
            compute_type=get_whisper_compute_type(),
        )
        transcribe_kw: dict = {"vad_filter": True}
        lang = get_whisper_language()
        if lang:
            transcribe_kw["language"] = lang
        segments, _ = model.transcribe(audio_path, **transcribe_kw)

        lines = []
        for idx, segment in enumerate(segments, start=1):
            text = str(segment.text).strip()
            if not text:
                continue
            lines.append(str(idx))
            lines.append(
                f"{self._format_srt_timestamp(segment.start)} --> {self._format_srt_timestamp(segment.end)}"
            )
            lines.append(text)
            lines.append("")

        srt_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.srt")
        with open(srt_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))
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

    def combine(self, image_paths: list[str], tts_path: str) -> tuple[str, str | None]:
        output_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.mp4")
        threads = get_threads()
        tts_clip = AudioFileClip(tts_path)
        max_duration = tts_clip.duration
        req_dur = max_duration / len(image_paths)
        out_w, out_h = get_video_output_size()
        target_ratio = out_w / out_h

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

        generator = lambda txt: TextClip(
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

        if get_verbose():
            info("[+] Combining image clips...")

        clips = []
        total_duration = 0.0
        while total_duration < max_duration:
            for image_path in image_paths:
                clip = ImageClip(image_path).with_duration(req_dur).with_fps(30)
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
                clips.append(clip.resized((out_w, out_h)))
                total_duration += req_dur
                if total_duration >= max_duration:
                    break

        final_clip = concatenate_videoclips(clips).with_fps(30)

        subtitle_path = None
        subtitles = None
        try:
            subtitle_path = self.generate_subtitles(tts_path)
            equalize_subtitles(subtitle_path, 10)
            subtitles = SubtitlesClip(subtitle_path, make_textclip=generator).with_position(
                ("center", subtitle_y)
            )
        except Exception as exc:
            warning(f"Subtitles unavailable, continuing without them: {exc}")

        bgm_path = choose_random_song()
        audio_tracks = [tts_clip.with_fps(44100)]
        if bgm_path:
            bgm = AudioFileClip(bgm_path).with_fps(44100).with_effects([afx.MultiplyVolume(0.1)])
            audio_tracks.append(bgm)
        final_audio = CompositeAudioClip(audio_tracks)
        final_clip = final_clip.with_audio(final_audio).with_duration(tts_clip.duration)

        if subtitles is not None:
            final_clip = CompositeVideoClip([final_clip, subtitles])

        final_clip.write_videofile(output_path, threads=threads)
        success(f'Wrote video to "{output_path}"')
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
