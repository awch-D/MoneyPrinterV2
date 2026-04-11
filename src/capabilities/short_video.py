from __future__ import annotations

from capabilities.base import RunContext
from pipeline.short_video_pipeline import ShortVideoPipeline, VideoBuildResult
from providers.image_api_provider import ImageApiProvider
from providers.script_api_provider import ScriptApiProvider


class ShortVideoCapability:
    name = "short"

    def run(self, ctx: RunContext) -> VideoBuildResult:
        pipeline = ShortVideoPipeline(
            script_provider=ScriptApiProvider(),
            image_provider=ImageApiProvider(),
        )
        return pipeline.run(
            niche=ctx.niche.strip(),
            language=ctx.language.strip(),
            topic=ctx.topic,
            script_file=ctx.script_file,
        )
