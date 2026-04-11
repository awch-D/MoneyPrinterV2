import json
import os
import sys
from contextlib import contextmanager
from typing import Any

import srt_equalizer
from termcolor import colored

ROOT_DIR = os.path.dirname(sys.path[0])

_RUNTIME_OVERRIDE_STACK: list[dict[str, Any]] = []


@contextmanager
def runtime_config_overrides(updates: dict[str, Any]):
    """Temporarily merge key/value pairs into config reads (does not write config.json)."""
    _RUNTIME_OVERRIDE_STACK.append(dict(updates))
    try:
        yield
    finally:
        _RUNTIME_OVERRIDE_STACK.pop()


def _read_config() -> dict[str, Any]:
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        cfg: dict[str, Any] = json.load(file)
    merged = dict(cfg)
    for layer in _RUNTIME_OVERRIDE_STACK:
        merged.update(layer)
    return merged


def assert_folder_structure() -> None:
    mp_dir = os.path.join(ROOT_DIR, ".mp")
    if not os.path.exists(mp_dir):
        if get_verbose():
            print(colored(f"=> Creating .mp folder at {mp_dir}", "green"))
        os.makedirs(mp_dir)


def get_first_time_running() -> bool:
    return not os.path.exists(os.path.join(ROOT_DIR, ".mp"))


def get_verbose() -> bool:
    return bool(_read_config().get("verbose", True))


def get_threads() -> int:
    return int(_read_config().get("threads", 2))


def get_zip_url() -> str:
    return str(_read_config().get("zip_url", "")).strip()


def get_tts_voice() -> str:
    return str(_read_config().get("tts_voice", "Jasper"))


def get_tts_backend() -> str:
    """kitten = local KittenTTS; qwen3_http / qwen3_gradio = Gradio HTTP API (e.g. local qwen3-tts app)."""
    return str(_read_config().get("tts_backend", "kitten")).strip().lower()


def get_qwen3_tts_url() -> str:
    return str(_read_config().get("qwen3_tts_url", "http://127.0.0.1:7862")).rstrip("/")


def get_qwen3_tts_api_name() -> str:
    """`/do_job_t` (text+instruct), `/do_job` (dropdown+ref audio), or `auto` (pick by reference_audio)."""
    return str(_read_config().get("qwen3_tts_api_name", "/do_job_t")).strip() or "/do_job_t"


def get_qwen3_tts_reference_audio() -> str:
    """Local wav/mp3 path for Gradio `/do_job` prompt_audio (required by that endpoint)."""
    return str(_read_config().get("qwen3_tts_reference_audio", "")).strip()


def get_qwen3_tts_voices_dropdown() -> str:
    return str(_read_config().get("qwen3_tts_voices_dropdown", "老男人")).strip() or "老男人"


def get_qwen3_tts_prompt_text() -> str:
    """Extra prompt for `/do_job`; empty means use qwen3_tts_instruct."""
    return str(_read_config().get("qwen3_tts_prompt_text", "")).strip()


def get_qwen3_tts_speed() -> float:
    return float(_read_config().get("qwen3_tts_speed", 1.0))


def get_qwen3_tts_batch() -> float:
    return float(_read_config().get("qwen3_tts_batch", 8))


def get_qwen3_tts_lang() -> str:
    return str(_read_config().get("qwen3_tts_lang", "Chinese")).strip() or "Chinese"


def get_qwen3_tts_model_type() -> str:
    raw = str(_read_config().get("qwen3_tts_model_type", "0.6B")).strip()
    return raw if raw in ("0.6B", "1.7B") else "0.6B"


def get_qwen3_tts_instruct() -> str:
    return str(_read_config().get("qwen3_tts_instruct", ""))


def get_qwen3_tts_max_chunk_chars() -> int:
    return int(_read_config().get("qwen3_tts_max_chunk_chars", 400))


def get_qwen3_tts_gradio_chunk_size() -> float:
    """Third argument to Gradio `/do_job_t`: predict(text, instruct, chunk_size, ...)."""
    return float(_read_config().get("qwen3_tts_gradio_chunk_size", 200))


def get_qwen3_tts_static_args() -> list:
    raw = _read_config().get("qwen3_tts_static_args", [])
    return list(raw) if isinstance(raw, list) else []


def get_qwen3_tts_predict_arg_order() -> list[str]:
    """Order of Gradio predict args after static_args, e.g. [\"instruct\", \"text\"] or [\"text\", \"instruct\"]."""
    raw = _read_config().get("qwen3_tts_predict_arg_order")
    if isinstance(raw, list) and raw:
        return [str(x).strip() for x in raw if str(x).strip()]
    return ["text", "instruct"]


def get_qwen3_tts_app_dir() -> str:
    """Directory containing the packaged `qwen3-tts` binary (run it manually before pipeline)."""
    return str(_read_config().get("qwen3_tts_app_dir", "")).strip()


def get_stt_provider() -> str:
    return str(_read_config().get("stt_provider", "local_whisper"))


def get_whisper_model() -> str:
    return str(_read_config().get("whisper_model", "base"))


def get_whisper_model_path() -> str:
    """If set, passed to openai-whisper CLI as ``--model`` (local checkpoint path or model id)."""
    return str(_read_config().get("whisper_model_path", "")).strip()


def get_whisper_language() -> str | None:
    """Optional ISO language code for transcribe(), e.g. zh — improves accuracy when fixed."""
    raw = str(_read_config().get("whisper_language", "")).strip()
    return raw or None


def get_whisper_device() -> str:
    return str(_read_config().get("whisper_device", "auto"))


def get_whisper_compute_type() -> str:
    return str(_read_config().get("whisper_compute_type", "int8"))


def get_whisper_cli_path() -> str:
    """Absolute path to the ``whisper`` executable, e.g. ``/opt/homebrew/bin/whisper``. Empty: use ``PATH``."""
    return str(_read_config().get("whisper_cli_path", "")).strip()


def get_whisper_cli_timeout_seconds() -> int:
    raw = _read_config().get("whisper_cli_timeout_seconds", 7200)
    try:
        v = int(raw)
    except (TypeError, ValueError):
        v = 7200
    return max(120, min(86400, v))


_WHISPER_CLI_MODEL_ALIASES: dict[str, str] = {
    # faster-whisper style name → openai-whisper CLI id (see ``whisper --help``)
    "large-v3-turbo": "turbo",
}


def get_whisper_model_for_cli() -> str:
    """Model id or local path for ``whisper`` CLI ``--model``."""
    mp = get_whisper_model_path().strip()
    if mp:
        return mp
    name = get_whisper_model().strip()
    return _WHISPER_CLI_MODEL_ALIASES.get(name.lower(), name)


def whisper_cli_uses_fp16() -> bool:
    """Map ``whisper_compute_type`` to openai-whisper ``--fp16``."""
    ct = get_whisper_compute_type().lower()
    return ct not in ("int8", "int8_float16", "int8_float32", "int8_bfloat16")


def whisper_cli_device_args() -> list[str]:
    dev = get_whisper_device().strip().lower()
    if not dev or dev == "auto":
        return []
    return ["--device", dev]


def resolve_whisper_cli_executable() -> str | None:
    import shutil

    p = get_whisper_cli_path().strip()
    if p and os.path.isfile(p):
        return p
    return shutil.which("whisper")


def get_assemblyai_api_key() -> str:
    return str(_read_config().get("assembly_ai_api_key", "")).strip()


def get_font() -> str:
    return str(_read_config().get("font", "bold_font.ttf"))


def get_subtitle_font_path() -> str:
    """Font file for burned-in subtitles (use a CJK .ttf/.ttc path for Chinese)."""
    raw = str(_read_config().get("subtitle_font", "")).strip()
    if raw and os.path.isfile(raw):
        return raw
    return os.path.join(get_fonts_dir(), get_font())


def get_subtitle_font_size_config() -> int | None:
    v = _read_config().get("subtitle_font_size")
    if isinstance(v, (int, float)) and v > 0:
        return int(v)
    return None


def get_subtitle_font_color() -> str:
    raw = str(_read_config().get("subtitle_font_color", "#FFFFFF")).strip()
    return raw or "#FFFFFF"


def get_subtitle_stroke_color() -> str | None:
    raw = _read_config().get("subtitle_stroke_color", "black")
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "none", "null")):
        return None
    return str(raw).strip()


def get_subtitle_stroke_width() -> int:
    return max(0, int(_read_config().get("subtitle_stroke_width", 3)))


def get_subtitle_bottom_margin() -> int:
    return max(0, int(_read_config().get("subtitle_bottom_margin", 52)))


def get_subtitle_caption_height_ratio() -> float:
    """Fraction of frame height reserved for the caption text box (bottom strip)."""
    r = float(_read_config().get("subtitle_caption_height_ratio", 0.22))
    return max(0.12, min(0.45, r))


def get_fonts_dir() -> str:
    return os.path.join(ROOT_DIR, "fonts")


def get_imagemagick_path() -> str:
    return str(_read_config().get("imagemagick_path", ""))


def get_script_sentence_length() -> int:
    return int(_read_config().get("script_sentence_length", 4))


def get_novel_chapter_max_segments() -> int:
    """Upper bound on LLM scene segments per chapter (cost/latency guard)."""
    return max(3, min(60, int(_read_config().get("novel_chapter_max_segments", 20))))


def get_image_prompt_style() -> str:
    """Full style block appended to every image prompt. If non-empty, overrides ``image_prompt_style_preset``."""
    return str(_read_config().get("image_prompt_style", "")).strip()


def get_image_prompt_style_preset() -> str:
    """
    Built-in preset key (see ``novel.image_style_presets.STYLE_PRESETS``) when ``image_prompt_style`` is empty.
    Use ``none`` to disable.
    """
    raw = _read_config().get("image_prompt_style_preset", "none")
    return str(raw).strip() if raw is not None else "none"


def get_script_api_base_url() -> str:
    return str(_read_config().get("script_api_base_url", "https://api.openai.com/v1")).rstrip("/")


def get_script_api_key() -> str:
    configured = str(_read_config().get("script_api_key", "")).strip()
    return configured or os.environ.get("SCRIPT_API_KEY", "").strip()


def get_script_api_model() -> str:
    return str(_read_config().get("script_api_model", "gpt-4.1-mini")).strip()


def get_nanobanana2_api_base_url() -> str:
    return str(
        _read_config().get(
            "nanobanana2_api_base_url", "https://generativelanguage.googleapis.com/v1beta"
        )
    ).rstrip("/")


def get_nanobanana2_api_key() -> str:
    configured = str(_read_config().get("nanobanana2_api_key", "")).strip()
    return configured or os.environ.get("GEMINI_API_KEY", "").strip()


def get_nanobanana2_model() -> str:
    return str(_read_config().get("nanobanana2_model", "gemini-3.1-flash-image"))


def get_nanobanana2_aspect_ratio() -> str:
    return str(_read_config().get("nanobanana2_aspect_ratio", "9:16"))


def get_nanobanana2_image_timeout_seconds() -> int:
    """HTTP read timeout for each image generation request (large models can be slow)."""
    return max(120, min(3600, int(_read_config().get("nanobanana2_image_timeout_seconds", 900))))


def get_nanobanana2_image_max_retries() -> int:
    return max(1, min(8, int(_read_config().get("nanobanana2_image_max_retries", 4))))


def get_video_output_size() -> tuple[int, int]:
    """Final frame size (width, height). Uses video_output_width/height if set, else video_output_aspect."""
    cfg = _read_config()
    w, h = cfg.get("video_output_width"), cfg.get("video_output_height")
    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return (w, h)
    aspect = str(cfg.get("video_output_aspect", "16:9")).strip().lower().replace("x", ":")
    if aspect in ("9:16", "portrait"):
        return (1080, 1920)
    # 16:9 and unknown values default to landscape HD
    return (1920, 1080)


def equalize_subtitles(srt_path: str, max_chars: int = 10) -> None:
    srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)
