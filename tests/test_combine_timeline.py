import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline.short_video_pipeline import ShortVideoPipeline


class _FakeScript:
    def generate_text(self, prompt: str) -> str:
        return ""


class _FakeImage:
    def generate_image_bytes(self, prompt: str, *, aspect_ratio: str | None = None) -> bytes:
        return b""


class CombineTimelineTests(unittest.TestCase):
    def test_length_mismatch_raises(self) -> None:
        pipe = ShortVideoPipeline(_FakeScript(), _FakeImage())
        with self.assertRaises(ValueError):
            pipe.combine_timeline(["only_one.png"], [1.0, 2.0], "/tmp/does_not_matter.wav")

    def test_empty_raises(self) -> None:
        pipe = ShortVideoPipeline(_FakeScript(), _FakeImage())
        with self.assertRaises(ValueError):
            pipe.combine_timeline([], [], "/tmp/x.wav")


if __name__ == "__main__":
    unittest.main()
