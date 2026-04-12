#!/usr/bin/env python3
"""
Preview novel-chapter segmentation only (script API → ChapterPlan JSON).

Does not call image API, TTS, or video. Use from repo root:

  PYTHONPATH=src python scripts/preview_chapter_segments.py --chapter-file stories/tianbao_short.txt

  PYTHONPATH=src python scripts/preview_chapter_segments.py --chapter-file chapter.txt --language Chinese --max-segments 8
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

from novel.chapter_analyzer import analyze_chapter  # noqa: E402
from novel.chapter_plan import ChapterPlan  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call chapter storyboard LLM only; print parsed segments as JSON."
    )
    parser.add_argument(
        "--chapter-file",
        required=True,
        help="Path to plain-text chapter (relative to cwd or absolute).",
    )
    parser.add_argument(
        "--language",
        default="Chinese",
        help="Voiceover language hint for the model (default: Chinese).",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional cap on segment count for this run only (default: no cap, same as novel_chapter pipeline).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Single-line JSON (no indent).",
    )
    args = parser.parse_args()

    path = args.chapter_file
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as f:
        text = f.read()

    provider = ScriptApiProvider()
    plan = analyze_chapter(
        text,
        args.language.strip(),
        provider,
        max_segments=args.max_segments,
    )

    out = _plan_to_dict(plan)
    out["_meta"] = {
        "chapter_file": path,
        "language": args.language.strip(),
        "segment_count": len(plan.segments),
        "character_count": len(plan.characters),
    }

    indent = None if args.compact else 2
    print(json.dumps(out, ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
