#!/usr/bin/env python3
import json
import os
import shutil
import sys
from typing import Tuple

import requests


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_url(url: str, timeout: int = 3) -> Tuple[bool, str]:
    try:
        response = requests.get(url, timeout=timeout)
        return True, f"HTTP {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    if not os.path.exists(CONFIG_PATH):
        fail(f"Missing config file: {CONFIG_PATH}")
        return 1

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    failures = 0

    stt_provider = str(cfg.get("stt_provider", "local_whisper")).lower()

    ok(f"stt_provider={stt_provider}")

    imagemagick_path = cfg.get("imagemagick_path", "")
    if imagemagick_path and os.path.exists(imagemagick_path):
        ok(f"imagemagick_path exists: {imagemagick_path}")
    else:
        warn(
            "imagemagick_path is not set to a valid executable path. "
            "MoviePy subtitle rendering may fail."
        )

    firefox_profile = cfg.get("firefox_profile", "")
    if firefox_profile:
        if os.path.isdir(firefox_profile):
            ok(f"firefox_profile exists: {firefox_profile}")
        else:
            warn(f"firefox_profile does not exist: {firefox_profile}")
    else:
        warn("firefox_profile is empty. Twitter/YouTube automation requires this.")

    # Script / storyboard LLM (OpenAI-compatible Chat Completions)
    script_base = str(cfg.get("script_api_base_url", "https://api.openai.com/v1")).rstrip("/")
    script_key = str(cfg.get("script_api_key", "") or os.environ.get("SCRIPT_API_KEY", "")).strip()
    script_model = str(cfg.get("script_api_model", "gpt-4.1-mini")).strip()

    if not script_base:
        fail("script_api_base_url is empty")
        failures += 1
    else:
        ok(f"script_api_base_url={script_base}")

    if script_key:
        ok("script_api_key is set (or SCRIPT_API_KEY in environment)")
    else:
        fail("script_api_key is empty (and SCRIPT_API_KEY is not set)")
        failures += 1

    if script_model:
        ok(f"script_api_model={script_model}")
    else:
        warn("script_api_model is empty; chat requests may fail at the gateway")

    reachable, detail = check_url(script_base, timeout=8)
    if not reachable:
        warn(f"script_api_base_url could not be reached: {detail}")
    else:
        ok(f"script API base URL reachable: {script_base}")

    # Nano Banana 2 (image generation)
    api_key = cfg.get("nanobanana2_api_key", "") or os.environ.get("GEMINI_API_KEY", "")
    nb2_base = str(
        cfg.get(
            "nanobanana2_api_base_url",
            "https://generativelanguage.googleapis.com/v1beta",
        )
    ).rstrip("/")
    if api_key:
        ok("nanobanana2_api_key is set")
    else:
        fail("nanobanana2_api_key is empty (and GEMINI_API_KEY is not set)")
        failures += 1

    reachable, detail = check_url(nb2_base, timeout=8)
    if not reachable:
        warn(f"Nano Banana 2 base URL could not be reached: {detail}")
    else:
        ok(f"Nano Banana 2 base URL reachable: {nb2_base}")

    if stt_provider == "local_whisper":
        cli = str(cfg.get("whisper_cli_path", "")).strip()
        whisper_bin = cli if cli and os.path.isfile(cli) else shutil.which("whisper")
        if not whisper_bin:
            fail(
                "Whisper CLI not found. Install openai-whisper (e.g. brew install openai-whisper) "
                "or set whisper_cli_path in config.json (e.g. /opt/homebrew/bin/whisper)."
            )
            failures += 1
        else:
            ok(f"Whisper CLI found: {whisper_bin}")

    if failures:
        print("")
        print(f"Preflight completed with {failures} blocking issue(s).")
        return 1

    print("")
    print("Preflight passed. Local setup looks ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
