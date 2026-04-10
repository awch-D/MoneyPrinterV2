import json
import os
import sys
from typing import Any

import srt_equalizer
from termcolor import colored

ROOT_DIR = os.path.dirname(sys.path[0])


def _read_config() -> dict[str, Any]:
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as file:
        return json.load(file)


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


def get_stt_provider() -> str:
    return str(_read_config().get("stt_provider", "local_whisper"))


def get_whisper_model() -> str:
    return str(_read_config().get("whisper_model", "base"))


def get_whisper_device() -> str:
    return str(_read_config().get("whisper_device", "auto"))


def get_whisper_compute_type() -> str:
    return str(_read_config().get("whisper_compute_type", "int8"))


def get_assemblyai_api_key() -> str:
    return str(_read_config().get("assembly_ai_api_key", "")).strip()


def get_font() -> str:
    return str(_read_config().get("font", "bold_font.ttf"))


def get_fonts_dir() -> str:
    return os.path.join(ROOT_DIR, "fonts")


def get_imagemagick_path() -> str:
    return str(_read_config().get("imagemagick_path", ""))


def get_script_sentence_length() -> int:
    return int(_read_config().get("script_sentence_length", 4))


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
    return str(_read_config().get("nanobanana2_model", "gemini-3.1-flash-image-preview"))


def get_nanobanana2_aspect_ratio() -> str:
    return str(_read_config().get("nanobanana2_aspect_ratio", "9:16"))


def equalize_subtitles(srt_path: str, max_chars: int = 10) -> None:
    srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)
