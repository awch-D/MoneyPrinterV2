"""Novel chapter analysis and scene segmentation for narrative video."""

from .chapter_plan import ChapterPlan, CharacterDef, SceneSegment
from .segment_schema import parse_chapter_plan_json

__all__ = ["ChapterPlan", "CharacterDef", "SceneSegment", "parse_chapter_plan_json"]
