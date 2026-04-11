from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CharacterDef:
    id: str
    name: str
    look: str


@dataclass
class SceneSegment:
    narration: str
    scene_summary: str
    image_prompt: str
    visible_character_ids: list[str] = field(default_factory=list)


@dataclass
class ChapterPlan:
    style_bible: str
    characters: list[CharacterDef]
    segments: list[SceneSegment]

    def full_script(self) -> str:
        return "\n".join(s.narration.strip() for s in self.segments if s.narration.strip())
