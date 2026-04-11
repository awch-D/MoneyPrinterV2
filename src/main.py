import argparse
import json
import sys

from art import print_banner
from config import assert_folder_structure, get_first_time_running
from pipeline.short_video_pipeline import ShortVideoPipeline
from providers.image_api_provider import ImageApiProvider
from providers.script_api_provider import ScriptApiProvider
from status import error, info
from utils import fetch_songs, rem_temp_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a short video with script -> images -> speech -> subtitles -> compose."
    )
    parser.add_argument(
        "--niche",
        default="",
        help="Niche used when topic is auto-generated (not required with --script-file).",
    )
    parser.add_argument("--language", default="English", help="Script language.")
    parser.add_argument("--topic", default="", help="Optional fixed topic. If omitted, generated from niche.")
    parser.add_argument(
        "--script-file",
        default="",
        help="Use novel/text file as narration: skips script & image APIs; placeholder background only.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not clear existing temp files in .mp before generation.",
    )
    args = parser.parse_args()
    if not args.script_file.strip() and not args.niche.strip():
        parser.error("--niche is required unless --script-file is set.")
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

    script_file = args.script_file.strip()
    pipeline = ShortVideoPipeline(
        script_provider=ScriptApiProvider(),
        image_provider=ImageApiProvider(),
    )
    result = pipeline.run(
        niche=args.niche.strip(),
        language=args.language.strip(),
        topic=(args.topic.strip() or None),
        script_file=script_file or None,
    )

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
