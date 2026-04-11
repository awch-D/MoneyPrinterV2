from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pipeline.short_video_pipeline import VideoBuildResult


@dataclass
class RunContext:
    niche: str
    language: str
    topic: str | None
    script_file: str | None
    chapter_file: str | None
    keep_temp: bool


class VideoCapability(Protocol):
    name: str

    def run(self, ctx: RunContext) -> VideoBuildResult: ...
