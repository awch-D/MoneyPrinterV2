#!/usr/bin/env python3
"""
Run every LLM step **before** the image API: no TTS, no Whisper, no video, no image requests.

Novel chapter (one storyboard JSON call, then local merge of prompts):

  PYTHONPATH=src python scripts/preview_pre_image.py novel_chapter \\
    --chapter-file stories/tianbao_short.txt --language Chinese

Short pipeline (topic → script → image prompt list):

  PYTHONPATH=src python scripts/preview_pre_image.py short --niche 科技 --language Chinese
  PYTHONPATH=src python scripts/preview_pre_image.py short --language Chinese --topic \"固定话题\"

Output is JSON on stdout. Put --compact on the **subcommand** (e.g. novel_chapter --compact ...).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import runtime_config_overrides  # noqa: E402
from novel.chapter_analyzer import analyze_chapter, build_merged_image_prompt  # noqa: E402
from novel.chapter_plan import ChapterPlan  # noqa: E402
from novel.image_style_presets import append_global_style_to_image_prompt  # noqa: E402
from pipeline.short_video_pipeline import ShortVideoPipeline  # noqa: E402
from providers.image_api_provider import ImageApiProvider  # noqa: E402
from providers.script_api_provider import ScriptApiProvider  # noqa: E402


def _plan_to_dict(plan: ChapterPlan) -> dict:
    return {
        "style_bible": plan.style_bible,
        "characters": [
            {"id": c.id, "name": c.name, "look": c.look} for c in plan.characters
        ],
        "segments": [
            {
                "narration": s.narration,
                "scene_summary": s.scene_summary,
                "image_prompt": s.image_prompt,
                "visible_character_ids": s.visible_character_ids,
            }
            for s in plan.segments
        ],
    }


def cmd_novel_chapter(args: argparse.Namespace) -> int:
    path = args.chapter_file
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as f:
        text = f.read()

    with runtime_config_overrides({"verbose": False}):
        provider = ScriptApiProvider()
        plan = analyze_chapter(
            text,
            args.language.strip(),
            provider,
            max_segments=args.max_segments,
        )

    topic_label = (args.topic or "").strip() or os.path.basename(path)
    merged: list[dict] = []
    for i, seg in enumerate(plan.segments, start=1):
        merged.append(
            {
                "segment_index": i,
                "merged_prompt_sent_to_image_api": build_merged_image_prompt(plan, seg),
            }
        )

    out = {
        "mode": "novel_chapter",
        "topic_label": topic_label,
        "chapter_file": path,
        "language": args.language.strip(),
        "plan": _plan_to_dict(plan),
        "merged_image_prompts": merged,
        "_meta": {
            "segment_count": len(plan.segments),
            "character_count": len(plan.characters),
            "note": "merged_prompt_* includes style_bible, character looks, scene context, segment image_prompt, and global image style preset.",
        },
    }
    indent = None if args.compact else 2
    print(json.dumps(out, ensure_ascii=False, indent=indent))
    return 0


def cmd_short(args: argparse.Namespace) -> int:
    niche = (args.niche or "").strip()
    topic_fix = (args.topic or "").strip()
    if not niche and not topic_fix:
        print("short: provide --niche and/or --topic (at least one).", file=sys.stderr)
        return 1

    with runtime_config_overrides({"verbose": False}):
        pipeline = ShortVideoPipeline(ScriptApiProvider(), ImageApiProvider())
        language = args.language.strip()
        resolved_topic = topic_fix or pipeline.generate_topic(niche, language)
        script = pipeline.generate_script(resolved_topic, language)
        raw_prompts = pipeline.generate_prompts(resolved_topic, script)

    prompts_out = [
        {
            "index": i,
            "raw_prompt_from_llm": p,
            "with_global_style": append_global_style_to_image_prompt(p),
        }
        for i, p in enumerate(raw_prompts, start=1)
    ]

    out = {
        "mode": "short",
        "niche": niche or None,
        "topic": resolved_topic,
        "language": language,
        "script": script,
        "image_prompts": prompts_out,
        "_meta": {
            "prompt_count": len(raw_prompts),
            "note": "Real pipeline sends with_global_style to the image API for each prompt.",
        },
    }
    indent = None if args.compact else 2
    print(json.dumps(out, ensure_ascii=False, indent=indent))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview LLM outputs before image generation (no image/TTS/video)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_novel = sub.add_parser(
        "novel_chapter",
        help="Chapter file → storyboard JSON + merged prompts per segment.",
    )
    p_novel.add_argument("--chapter-file", required=True)
    p_novel.add_argument("--language", default="Chinese")
    p_novel.add_argument("--topic", default="", help="Label only (for JSON metadata).")
    p_novel.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional cap for testing (default: no cap).",
    )
    p_novel.add_argument(
        "--compact",
        action="store_true",
        help="Single-line JSON.",
    )
    p_novel.set_defaults(func=cmd_novel_chapter)

    p_short = sub.add_parser(
        "short",
        help="Niche/topic → script → image prompt list (as in short pipeline).",
    )
    p_short.add_argument("--niche", default="")
    p_short.add_argument("--topic", default="")
    p_short.add_argument("--language", default="English")
    p_short.add_argument(
        "--compact",
        action="store_true",
        help="Single-line JSON.",
    )
    p_short.set_defaults(func=cmd_short)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
