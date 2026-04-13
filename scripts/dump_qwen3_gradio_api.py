#!/usr/bin/env python3
"""Print Gradio ``view_api()`` for ``qwen3_tts_url`` — verify /do_job exists and arg order."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import get_qwen3_tts_url  # noqa: E402


def main() -> int:
    from gradio_client import Client

    url = get_qwen3_tts_url()
    print("URL:", url)
    try:
        c = Client(url)
    except Exception as e:  # noqa: BLE001
        print("ERROR: cannot connect:", e, file=sys.stderr)
        print("Start qwen3-tts / Gradio first; check qwen3_tts_url in config.json.", file=sys.stderr)
        return 1
    print(c.view_api())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
