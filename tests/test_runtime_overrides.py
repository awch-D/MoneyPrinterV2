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


class RuntimeOverridesTests(unittest.TestCase):
    def write_minimal_config(self, directory: str) -> None:
        payload = {
            "verbose": False,
            "video_output_aspect": "16:9",
            "nanobanana2_aspect_ratio": "16:9",
        }
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def test_orientation_overrides_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_minimal_config(temp_dir)
            with patch.object(config, "ROOT_DIR", temp_dir):
                self.assertEqual(config.get_video_output_size(), (1920, 1080))
                with config.runtime_config_overrides(
                    {"video_output_aspect": "9:16", "nanobanana2_aspect_ratio": "9:16"}
                ):
                    self.assertEqual(config.get_video_output_size(), (1080, 1920))
                    self.assertEqual(config.get_nanobanana2_aspect_ratio(), "9:16")
                self.assertEqual(config.get_video_output_size(), (1920, 1080))


if __name__ == "__main__":
    unittest.main()
