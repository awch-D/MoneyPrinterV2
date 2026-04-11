from __future__ import annotations

import json
import re
from typing import Any

from .chapter_plan import ChapterPlan, CharacterDef, SceneSegment


def _parse_json_root(blob: str) -> dict[str, Any]:
    """Strict json.loads, then json-repair fallback for common LLM mistakes (e.g. raw newlines in strings)."""
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as first_exc:
        try:
            from json_repair import repair_json
        except ImportError as imp_exc:
            raise first_exc from imp_exc
        try:
            repaired = repair_json(blob, return_objects=True)
            if isinstance(repaired, dict):
                data = repaired
            else:
                data = json.loads(repair_json(blob))
        except Exception as repair_exc:
            raise first_exc from repair_exc
    if not isinstance(data, dict):
        raise ValueError("chapter plan: JSON root must be an object")
    return data


def _extract_json_object(raw: str) -> str:
    """Take the first complete top-level `{...}` using brace depth (strings-aware)."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced JSON braces in model output")


def parse_chapter_plan_json(raw: str) -> ChapterPlan:
    """Parse strict chapter plan JSON from LLM output."""
    blob = _extract_json_object(raw)
    data = _parse_json_root(blob)

    style_bible = str(data.get("style_bible", "")).strip()
    if not style_bible:
        raise ValueError("chapter plan: missing style_bible")

    characters: list[CharacterDef] = []
    for item in data.get("characters") or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        look = str(item.get("look", "")).strip()
        if cid and name and look:
            characters.append(CharacterDef(id=cid, name=name, look=look))

    segments: list[SceneSegment] = []
    for item in data.get("segments") or []:
        if not isinstance(item, dict):
            continue
        narration = str(item.get("narration", "")).strip()
        scene_summary = str(item.get("scene_summary", "")).strip()
        image_prompt = str(item.get("image_prompt", "")).strip()
        vids = item.get("visible_character_ids")
        if not isinstance(vids, list):
            vids = []
        visible = [str(x).strip() for x in vids if str(x).strip()]
        if narration and image_prompt:
            segments.append(
                SceneSegment(
                    narration=narration,
                    scene_summary=scene_summary,
                    image_prompt=image_prompt,
                    visible_character_ids=visible,
                )
            )

    if not segments:
        raise ValueError("chapter plan: no valid segments")

    return ChapterPlan(style_bible=style_bible, characters=characters, segments=segments)
