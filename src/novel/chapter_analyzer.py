from __future__ import annotations

import json

from config import get_novel_chapter_max_segments, get_verbose
from .chapter_plan import ChapterPlan, SceneSegment
from .image_style_presets import append_global_style_to_image_prompt
from .segment_schema import parse_chapter_plan_json
from providers.script_api_provider import ScriptApiProvider
from status import info, warning


def build_merged_image_prompt(plan: ChapterPlan, seg: SceneSegment) -> str:
    """Prepend style bible and visible character looks for visual consistency."""
    parts: list[str] = [plan.style_bible]
    by_id = {c.id: c for c in plan.characters}
    for cid in seg.visible_character_ids:
        ch = by_id.get(cid)
        if ch:
            parts.append(f"{ch.name}: {ch.look}")
    if seg.scene_summary:
        parts.append(f"Scene context: {seg.scene_summary}")
    parts.append(seg.image_prompt)
    return append_global_style_to_image_prompt("\n".join(parts))


def _analysis_prompt(chapter_text: str, language: str, max_segments: int) -> str:
    return f"""You are a storyboard director for ONE episode = ONE chapter of a narrative novel.

Task:
1) Read the chapter (may be long). Split it into {max_segments} or fewer visual scenes.
2) Each scene must have: narration (spoken voiceover in {language}), one cinematic image_prompt, and which characters appear.
3) Keep a consistent visual style across the episode.

Output rules:
- Return ONLY a single JSON object, no markdown fences, no commentary.
- CRITICAL: Every string value must stay on ONE line inside the JSON. Do not put raw line breaks inside double-quoted strings (that breaks parsers). If narration needs a line break, use the two characters backslash-n as \\n inside the string.
- JSON schema:
{{
  "style_bible": "short paragraph: art style, era, color palette, camera language",
  "characters": [
    {{"id": "c1", "name": "...", "look": "fixed hair/clothes/body markers; do not drift between scenes"}}
  ],
  "segments": [
    {{
      "narration": "paragraph or sentences for voiceover this beat",
      "scene_summary": "one line what happens",
      "image_prompt": "English cinematic description; must match narration; include lighting and composition",
      "visible_character_ids": ["c1"]
    }}
  ]
}}

Constraints:
- segments.length between 3 and {max_segments}.
- Every image_prompt must reflect its narration for that same segment.
- visible_character_ids must reference existing character id values.
- If the chapter names no people, invent minimal character ids for recurring figures.

Chapter text:
---
{chapter_text}
---
"""


def _repair_json_prompt(broken_snippet: str) -> str:
    return (
        "The following text was supposed to be ONLY one JSON object for a chapter storyboard "
        "with keys style_bible, characters, segments. Fix it to valid JSON only, no markdown.\n\n"
        f"{broken_snippet[:12000]}"
    )


def analyze_chapter(
    chapter_text: str,
    language: str,
    script_provider: ScriptApiProvider,
    *,
    max_segments: int | None = None,
) -> ChapterPlan:
    cap = max_segments if max_segments is not None else get_novel_chapter_max_segments()
    prompt = _analysis_prompt(chapter_text.strip(), language, cap)
    if get_verbose():
        info("Calling script API for chapter storyboard (JSON)...")
    raw = script_provider.generate_text(prompt, json_object=True)
    plan: ChapterPlan | None = None
    last_err: Exception | None = None
    try:
        plan = parse_chapter_plan_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        last_err = exc
        warning(f"Chapter JSON parse failed ({exc!s}); asking model to repair (up to 3 attempts)...")
        snippet = raw
        for attempt in range(3):
            snippet = script_provider.generate_text(_repair_json_prompt(snippet))
            try:
                plan = parse_chapter_plan_json(snippet)
                break
            except (json.JSONDecodeError, ValueError) as exc2:
                last_err = exc2
                warning(f"Repair attempt {attempt + 1}/3 failed: {exc2!s}")
        if plan is None:
            raise RuntimeError(f"Chapter storyboard JSON invalid after repairs: {last_err!s}") from last_err

    if len(plan.segments) > cap:
        warning(f"Chapter has {len(plan.segments)} segments; truncating to {cap}.")
        plan = ChapterPlan(
            style_bible=plan.style_bible,
            characters=plan.characters,
            segments=plan.segments[:cap],
        )
    return plan
