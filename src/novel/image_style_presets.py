"""Built-in image prompt style presets; combined with config ``image_prompt_style`` / ``image_prompt_style_preset``."""

from __future__ import annotations

STYLE_PRESETS: dict[str, str] = {
    "none": "",
    "han_guofeng_woodcut": (
        "杰作，最佳画质，8k，超精细细节，电影级镜头，中国古代汉代历史题材，新中式国风插画，"
        "复古木刻版画肌理，粗粝手绘笔触，厚重厚涂质感，暗黑悲凉氛围，肃杀压抑的历史叙事感，"
        "电影级布光，高对比度，冷青色调与暗血色撞色，低饱和度暗调基底，斑驳风化的材质纹理，"
        "画面做旧磨损效果，纸张颗粒质感，大景深，戏剧化阴影，广角镜头，极强的空间层次感，"
        "静态画面带有强烈的故事张力，画面暗角，复古动画电影美学"
    ),
    "cinematic_clean": (
        "masterpiece, best quality, 8k, ultra detailed, cinematic lighting, "
        "professional color grading, shallow depth of field, subtle film grain"
    ),
}


def list_style_preset_keys() -> list[str]:
    return sorted(k for k in STYLE_PRESETS if k != "none")


def resolve_global_style_text() -> str:
    from config import get_image_prompt_style, get_image_prompt_style_preset
    from status import warning

    custom = get_image_prompt_style().strip()
    if custom:
        return custom
    key = (get_image_prompt_style_preset() or "none").strip().lower()
    if not key or key == "none":
        return ""
    if key not in STYLE_PRESETS:
        warning(
            f"Unknown image_prompt_style_preset={key!r}; "
            f"valid: none, {', '.join(list_style_preset_keys())}. No extra style applied."
        )
        return ""
    return STYLE_PRESETS[key].strip()


def append_global_style_to_image_prompt(prompt: str) -> str:
    style = resolve_global_style_text()
    if not style:
        return prompt
    return f"{prompt.rstrip()}\n\n【全局画风】\n{style}"
