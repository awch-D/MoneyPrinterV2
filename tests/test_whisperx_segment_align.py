"""Unit tests for WhisperX segment duration helpers (no whisperx import required)."""

from __future__ import annotations

import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from novel.whisperx_segment_align import (  # noqa: E402
    _durations_from_words,
    fallback_proportional_segment_durations,
)


class WhisperxSegmentAlignTests(unittest.TestCase):
    def test_fallback_proportional(self) -> None:
        d = fallback_proportional_segment_durations(["a", "bbb"], 4.0)
        self.assertEqual(len(d), 2)
        self.assertAlmostEqual(sum(d), 4.0, places=5)

    def test_durations_from_words_two_segments(self) -> None:
        words = [
            {"word": "甲", "start": 0.0, "end": 0.5},
            {"word": "乙", "start": 0.5, "end": 1.0},
            {"word": "丙", "start": 1.0, "end": 2.0},
        ]
        segment_texts = ["甲乙", "丙"]
        full_ref = "甲乙丙"
        d = _durations_from_words(words, segment_texts, full_ref)
        self.assertEqual(len(d), 2)
        self.assertAlmostEqual(d[0], 1.0, places=3)
        self.assertAlmostEqual(d[1], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
