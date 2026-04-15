"""Built-in image prompt style presets; combined with config ``image_prompt_style`` / ``image_prompt_style_preset``."""

from __future__ import annotations

STYLE_PRESETS: dict[str, str] = {
    "none": "",
    "han_guofeng_woodcut": (
        "graphic novel style, bold black outlines, cel-shaded flat color blocks with painterly brush texture, "
        "weathered aged surface textures, dark oxblood maroon red and muted cyan teal-blue as dual dominant colors "
        "with dark red slightly more, the cool tone is cyan teal-blue NOT green, aged peeling vermillion tones NOT vivid bright red, "
        "extreme chiaroscuro lighting, moody dark atmospheric tone, semi-realistic mature anime illustration, cinematic composition"
    ),
    "han_guofeng_woodcut_strong": (
        "CRITICAL STYLE REQUIREMENTS - MUST FOLLOW:\n"
        "Art Style: graphic novel style, bold black outlines, cel-shaded flat color blocks with painterly brush texture, "
        "weathered aged surface textures, semi-realistic mature anime illustration.\n"
        "Color Palette: dark oxblood maroon red and muted cyan teal-blue as dual dominant colors with dark red slightly more. "
        "IMPORTANT: the cool tone is cyan teal-blue NOT green. Aged peeling vermillion tones NOT vivid bright red.\n"
        "Lighting: extreme chiaroscuro lighting, moody dark atmospheric tone.\n"
        "Composition: cinematic composition, dramatic framing.\n"
        "Quality: masterpiece, best quality, 8K ultra-detailed.\n"
        "FORBIDDEN: No bright green tones, no vivid bright red, no modern clean digital look, no cartoon style."
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
    # 将全局风格放在最前面，增加权重
    return f"【全局画风 - 必须遵循】\n{style}\n\n【场景描述】\n{prompt.rstrip()}"
