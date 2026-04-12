from __future__ import annotations

import json

from config import get_novel_chapter_image_prompt_suffix, get_verbose
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
    merged = "\n".join(parts)
    suffix = get_novel_chapter_image_prompt_suffix()
    if suffix:
        merged = f"{merged}\n{suffix}"
    return append_global_style_to_image_prompt(merged)


def _analysis_prompt(chapter_text: str, language: str, max_segments: int | None) -> str:
    if max_segments is not None:
        segment_rules = f"""1) Read the chapter. Split into **at most {max_segments}** segments while covering the narrative as fully as possible within that budget—do not drop the ending.
2) Prefer **shot-level** beats (one image + one short VO each), not long plot chunks. If the budget is tight, merge only **adjacent micro-beats**; never collapse unrelated events into one segment.
3) Each segment's VO should still target **5–8 seconds** at normal pace in {language} when possible; only merge beats when needed to stay within the segment budget."""
        constraints = f"""- segments.length must be between 3 and {max_segments} (inclusive)."""
    else:
        segment_rules = f"""1) Read the **full** chapter from start to finish. Break it into **shot-level** segments: each segment = **one still image** and **one voiceover clip** for a **single narrative beat** (one turn in action, emotion, revelation, or locale)—like a cut in film. Do **not** merge multiple beats into one segment to reduce count.
2) **Narration length (strict, audio–picture sync)**: Each segment's narration must target **5–8 seconds** at normal speaking pace in {language}. For **Chinese**, assume roughly **3–4 characters per second**—so usually **one short sentence** and typically **under ~28 characters** per segment. If a source sentence is longer, packs several clauses, or would exceed ~8s when read aloud, **split** into consecutive segments. **Never** use one segment for a whole novel paragraph or a multi-beat block that would read longer than ~8 seconds.
3) **1 beat = 1 visual idea**: If one written sentence combines several clauses (e.g. time/place + event + reaction), split so each segment shows **one clear still** the listener can hold for 5–8s. Prefer **finer** cuts over “plot-level” lumps where one image must cover 10–18s of VO (that causes audio–video drift in short-form video).
4) **Camera progression**: within a dramatic sequence use **establishing ultra-wide → wide → medium → close / tight** as tension rises. Each image_prompt must **name the shot scale** in English (e.g. ultra-wide establishing, high-angle wide, medium shot, medium close-up, tight close-up, extreme close-up) plus lighting and composition.
5) **Time / place jumps**: when the story jumps in time or geography (e.g. border to capital days later), use **separate segments** before and after the jump; in image_prompt note **transition intent** (e.g. hard cut, cross-dissolve / cross-fade mood) so the edit reads clearly.
6) **Visual anchors**: keep a coherent palette across shots (e.g. deep indigo night, amber torchlight, distant beacon orange) for continuity.
7) **Completeness**: cover every important beat from the source. **Do not mirror the chapter's paragraph breaks** if that yields long VO—subdivide **inside** paragraphs. When unsure whether to split, **split** for tighter sync."""
        constraints = """- Minimum 3 segments; **no upper limit**—use as many shots as needed so **no** segment's VO is routinely longer than ~8 seconds at normal pace."""

    tail = f"""A) Each segment: narration (voiceover in {language}), scene_summary, image_prompt (English), visible_character_ids.
B) style_bible: one short paragraph binding art era, palette, and camera language (e.g. wide for scope, push-in for tension, close-ups for emotion).
C) scene_summary: one non-empty line—what **this shot** shows (beat + framing role), not a whole scene synopsis.
D) image_prompt: English only; must match **only** that segment's narration; include shot scale, lighting, key subjects, and where relevant transition hints (e.g. cross-dissolve after a time jump).
E) characters[]: every **named or recurring** on-screen figure gets an id + fixed look; crowds without lines may omit ids (empty visible_character_ids)."""

    return f"""You are a storyboard director for ONE episode = ONE chapter of a narrative novel.

Task:
{segment_rules}
{tail}

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
      "narration": "one short sentence for this shot only; target 5-8s VO in the output language",
      "scene_summary": "one line: this shot's beat and framing role",
      "image_prompt": "English: shot scale + cinematic description + lighting; match narration only",
      "visible_character_ids": ["c1"]
    }}
  ]
}}

Constraints:
{constraints}
- Every image_prompt must reflect its narration for that same segment only.
- visible_character_ids must reference existing character id values (use [] if no named character).
- Register characters for all recurring named figures who appear on screen.

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
    """Split chapter into scenes via LLM. By default there is **no** segment cap; pass ``max_segments`` only for tests or manual budgets."""
    prompt = _analysis_prompt(chapter_text.strip(), language, max_segments)
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

    return plan
