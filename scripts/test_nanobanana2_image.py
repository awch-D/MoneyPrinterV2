#!/usr/bin/env python3
"""
Smoke-test Gemini/OpenAI-compatible image API (same contract as Novel2Toon gemini-image-gen skill).

Run from repo root:
  PYTHONPATH=src python3 scripts/test_nanobanana2_image.py

Uses nanobanana2_* keys from config.json (or pass --prompt).
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo root = parent of scripts/
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

from providers.image_api_provider import ImageApiProvider  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="A red lantern in snow at night, cinematic, 16:9 composition.",
        help="Test prompt",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write bytes (extension auto from magic bytes if omitted)",
    )
    args = parser.parse_args()

    os.chdir(_REPO)
    prov = ImageApiProvider()
    print("[test] Calling ImageApiProvider.generate_image_bytes ...")
    data = prov.generate_image_bytes(args.prompt)
    print(f"[test] OK, received {len(data)} bytes")

    if args.output:
        out = args.output
    else:
        ext = ".jpg" if data[:3] == b"\xff\xd8\xff" else ".png"
        out = os.path.join(_REPO, ".mp", f"smoke_image{ext}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
    base, old_ext = os.path.splitext(out)
    if data[:3] == b"\xff\xd8\xff":
        out = base + ".jpg"
    elif data[:4] == b"\x89PNG":
        out = base + ".png"
    with open(out, "wb") as f:
        f.write(data)
    print(f"[test] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
