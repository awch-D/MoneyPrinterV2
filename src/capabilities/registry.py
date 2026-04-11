from __future__ import annotations

from capabilities.base import RunContext, VideoCapability
from capabilities.novel_chapter import NovelChapterCapability
from capabilities.short_video import ShortVideoCapability

_CAPABILITIES: dict[str, VideoCapability] = {
    "short": ShortVideoCapability(),
    "novel_chapter": NovelChapterCapability(),
}

CAPABILITY_NAMES: tuple[str, ...] = tuple(sorted(_CAPABILITIES.keys()))


def get_capability(name: str) -> VideoCapability:
    key = (name or "").strip().lower()
    if key not in _CAPABILITIES:
        raise ValueError(f"Unknown capability {name!r}; choose one of: {', '.join(CAPABILITY_NAMES)}")
    return _CAPABILITIES[key]


def run_capability(name: str, ctx: RunContext) -> VideoBuildResult:
    return get_capability(name).run(ctx)
