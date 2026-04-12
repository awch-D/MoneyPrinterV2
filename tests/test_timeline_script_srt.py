"""Unit tests for per-segment script SRT (novel timeline subtitles)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from novel.timeline_script_srt import (  # noqa: E402
    seconds_to_srt_timestamp,
    write_timeline_script_subtitles_srt,
)


class TimelineScriptSrtTests(unittest.TestCase):
    def test_seconds_to_srt_timestamp_zero(self) -> None:
        self.assertEqual(seconds_to_srt_timestamp(0), "00:00:00,000")

    def test_seconds_to_srt_timestamp_with_ms(self) -> None:
        self.assertEqual(seconds_to_srt_timestamp(3.58), "00:00:03,580")

    def test_seconds_to_srt_timestamp_minute_carry(self) -> None:
        self.assertEqual(seconds_to_srt_timestamp(65.432), "00:01:05,432")

    def test_write_timeline_script_subtitles_srt(self) -> None:
        tf = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
        path = tf.name
        tf.close()
        try:
            out = write_timeline_script_subtitles_srt(
                ["第一句。", "第二句更长一点。"],
                [1.0, 2.5],
                srt_path=path,
            )
            self.assertEqual(out, os.path.abspath(path))
            with open(path, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("00:00:00,000 --> 00:00:01,000", text)
            self.assertIn("00:00:01,000 --> 00:00:03,500", text)
            self.assertIn("第一句。", text)
            self.assertIn("第二句更长一点。", text)
            self.assertTrue(text.strip().startswith("1\n"))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_write_timeline_script_subtitles_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            write_timeline_script_subtitles_srt(["a"], [1.0, 2.0], srt_path="/tmp/x.srt")


if __name__ == "__main__":
    unittest.main()
