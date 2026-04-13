#!/usr/bin/env python3
"""One-shot Qwen3 Gradio TTS test using config.json (reference audio + /do_job when auto)."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from classes.Tts import TTS  # noqa: E402
from config import (  # noqa: E402
    ROOT_DIR,
    get_qwen3_tts_reference_audio,
    get_qwen3_tts_voices_dropdown,
    runtime_config_overrides,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--text",
        default="这是参考音色测试。历史叙事，语气沉稳。",
        help="Text to synthesize",
    )
    p.add_argument(
        "-o",
        "--output",
        default="",
        help="Output WAV path (default: .mp/tts_reference_test.wav)",
    )
    p.add_argument(
        "--voices",
        default="",
        help="Override qwen3_tts_voices_dropdown for this run only (e.g. 使用参考音频)",
    )
    args = p.parse_args()

    out = args.output or os.path.join(ROOT_DIR, ".mp", "tts_reference_test.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ref = get_qwen3_tts_reference_audio()
    if not ref or not os.path.isfile(ref):
        print("ERROR: qwen3_tts_reference_audio missing or not a file:", ref, file=sys.stderr)
        return 1

    overrides: dict = {}
    if args.voices.strip():
        overrides["qwen3_tts_voices_dropdown"] = args.voices.strip()

    def _run() -> None:
        print("reference_audio:", ref)
        print("voices_dropdown:", get_qwen3_tts_voices_dropdown())
        print("synthesizing ->", out)
        TTS().synthesize(args.text, out)

    if overrides:
        with runtime_config_overrides(overrides):
            _run()
    else:
        _run()

    if not os.path.isfile(out):
        print("ERROR: output not written", file=sys.stderr)
        return 1
    print("OK", out, os.path.getsize(out), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
