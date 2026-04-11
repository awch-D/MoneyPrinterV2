import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from novel import image_style_presets


class ImageStylePresetsTests(unittest.TestCase):
    def write_config(self, directory: str, extra: dict) -> None:
        payload = {
            "verbose": False,
            "video_output_aspect": "16:9",
            "nanobanana2_aspect_ratio": "16:9",
        }
        payload.update(extra)
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def test_preset_han_appends_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_config(
                temp_dir,
                {"image_prompt_style": "", "image_prompt_style_preset": "han_guofeng_woodcut"},
            )
            with patch.object(config, "ROOT_DIR", temp_dir):
                out = image_style_presets.append_global_style_to_image_prompt("A warrior at dawn.")
                self.assertIn("A warrior at dawn.", out)
                self.assertIn("【全局画风】", out)
                self.assertIn("汉代", out)

    def test_custom_style_overrides_preset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_config(
                temp_dir,
                {
                    "image_prompt_style": "CUSTOM_STYLE_TAG",
                    "image_prompt_style_preset": "han_guofeng_woodcut",
                },
            )
            with patch.object(config, "ROOT_DIR", temp_dir):
                out = image_style_presets.append_global_style_to_image_prompt("scene")
                self.assertIn("CUSTOM_STYLE_TAG", out)
                self.assertNotIn("木刻", out)

    def test_none_preset_no_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_config(temp_dir, {"image_prompt_style_preset": "none"})
            with patch.object(config, "ROOT_DIR", temp_dir):
                out = image_style_presets.append_global_style_to_image_prompt("only scene")
                self.assertEqual(out, "only scene")


if __name__ == "__main__":
    unittest.main()
