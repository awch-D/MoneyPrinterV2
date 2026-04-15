import argparse
import json
import os
import sys

# 全局禁用代理 - 必须在所有其他导入之前
import disable_proxy  # noqa: F401

from art import print_banner
from capabilities.base import RunContext
from capabilities.registry import CAPABILITY_NAMES, run_capability
from config import assert_folder_structure, get_first_time_running, runtime_config_overrides
from status import error, info
from utils import fetch_songs, rem_temp_files


def _orientation_config(orientation: str) -> dict[str, str]:
    o = (orientation or "landscape").strip().lower()
    if o == "portrait":
        return {"video_output_aspect": "9:16", "nanobanana2_aspect_ratio": "9:16"}
    return {"video_output_aspect": "16:9", "nanobanana2_aspect_ratio": "16:9"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoneyPrinterV2: short video or novel-chapter narrative pipeline."
    )
    parser.add_argument(
        "--capability",
        default="short",
        choices=list(CAPABILITY_NAMES),
        help="short = topic/script pipeline; novel_chapter = one chapter → timed scenes + images.",
    )
    parser.add_argument(
        "--niche",
        default="",
        help="Niche when topic is auto-generated (short capability; not required with --script-file).",
    )
    parser.add_argument("--language", default="English", help="Script / narration language hint for the LLM and TTS.")
    parser.add_argument("--topic", default="", help="Optional fixed topic or chapter title label.")
    parser.add_argument(
        "--script-file",
        default="",
        help="(short) Use text file as narration; skips script & image APIs; placeholder visuals.",
    )
    parser.add_argument(
        "--chapter-file",
        default="",
        help="(novel_chapter) Path to one chapter of plain text (one episode per file).",
    )
    parser.add_argument(
        "--orientation",
        default="landscape",
        choices=["landscape", "portrait"],
        help="Output and image aspect: landscape 16:9 (default) or portrait 9:16.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not clear existing temp files in .mp before generation.",
    )
    parser.add_argument(
        "--reuse-images-manifest",
        action="store_true",
        help="(novel_chapter) Skip image generation and reuse image_paths from .mp/last_timeline_manifest.json.",
    )
    parser.add_argument(
        "--placeholder-images",
        action="store_true",
        help="(novel_chapter) Skip image generation and use placeholder images for every segment.",
    )
    args = parser.parse_args()

    cap = args.capability.strip().lower()
    if cap == "short":
        if not args.script_file.strip() and not args.niche.strip():
            parser.error("--niche is required for short capability unless --script-file is set.")
    elif cap == "novel_chapter":
        if not args.chapter_file.strip():
            parser.error("--chapter-file is required for novel_chapter capability.")

    return args


def main() -> int:
    args = parse_args()
    print_banner()

    if get_first_time_running():
        info("First run detected. Initializing workspace.")

    assert_folder_structure()
    if not args.keep_temp:
        rem_temp_files()
    fetch_songs()

    ctx = RunContext(
        niche=args.niche.strip(),
        language=args.language.strip(),
        topic=(args.topic.strip() or None),
        script_file=(args.script_file.strip() or None),
        chapter_file=(args.chapter_file.strip() or None),
        keep_temp=bool(args.keep_temp),
        reuse_images_manifest=bool(args.reuse_images_manifest),
        placeholder_images=bool(args.placeholder_images),
    )

    overrides = _orientation_config(args.orientation)
    with runtime_config_overrides(overrides):
        result = run_capability(args.capability, ctx)

    print(
        json.dumps(
            {
                "topic": result.topic,
                "script": result.script,
                "video_path": result.video_path,
                "audio_path": result.audio_path,
                "subtitle_path": result.subtitle_path,
                "image_paths": result.image_paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        error(f"Generation failed: {exc}")
        raise SystemExit(1) from exc
