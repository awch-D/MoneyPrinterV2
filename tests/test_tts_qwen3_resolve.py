"""Tests for Qwen3 Gradio API name resolution (no Gradio server required)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from classes.Tts import TTS  # noqa: E402


class Qwen3NoSentenceSplitTests(unittest.TestCase):
    def test_no_sentence_split_single_predict(self) -> None:
        """full_track path must not split on 。 when qwen3_no_sentence_split=True."""
        with patch("classes.Tts.get_tts_backend", return_value="qwen3"):
            tts = TTS()
            with patch.object(tts, "_predict_qwen3", return_value="/fake/out.wav") as pred:
                with patch("classes.Tts._normalize_gradio_output", return_value="/fake/out.wav"):
                    with patch("classes.Tts.shutil.copy2"):
                        tts.synthesize(
                            "第一句。第二句。第三句。",
                            "/tmp/x.wav",
                            qwen3_no_sentence_split=True,
                        )
            self.assertEqual(pred.call_count, 1)

    def test_default_splits_by_sentence(self) -> None:
        with patch("classes.Tts.get_tts_backend", return_value="qwen3"):
            tts = TTS()
            with patch.object(tts, "_predict_qwen3", return_value="/fake/out.wav") as pred:
                with patch("classes.Tts._normalize_gradio_output", return_value="/fake/out.wav"):
                    with patch("classes.Tts.sf.read") as sfr:
                        sfr.return_value = (__import__("numpy").zeros(100, dtype="float32"), 24000)
                        with patch("classes.Tts.sf.write"):
                            tts.synthesize("第一句。第二句。第三句。", "/tmp/y.wav")
            self.assertEqual(pred.call_count, 3)


class Qwen3ResolveApiNameTests(unittest.TestCase):
    def test_auto_without_reference_uses_do_job_t(self) -> None:
        with (
            patch("classes.Tts.get_tts_backend", return_value="qwen3"),
            patch("classes.Tts.get_qwen3_tts_api_name", return_value="auto"),
            patch("classes.Tts.get_qwen3_tts_reference_audio", return_value=""),
            patch("classes.Tts.get_verbose", return_value=False),
        ):
            tts = TTS()
            self.assertEqual(tts._resolve_qwen3_api_name(), "/do_job_t")

    def test_auto_with_reference_file_uses_do_job(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with (
                patch("classes.Tts.get_tts_backend", return_value="qwen3"),
                patch("classes.Tts.get_qwen3_tts_api_name", return_value="auto"),
                patch("classes.Tts.get_qwen3_tts_reference_audio", return_value=path),
                patch("classes.Tts.get_verbose", return_value=False),
            ):
                tts = TTS()
                self.assertEqual(tts._resolve_qwen3_api_name(), "/do_job")
        finally:
            os.unlink(path)

    def test_explicit_do_job_without_reference_raises(self) -> None:
        with (
            patch("classes.Tts.get_tts_backend", return_value="qwen3"),
            patch("classes.Tts.get_qwen3_tts_api_name", return_value="/do_job"),
            patch("classes.Tts.get_qwen3_tts_reference_audio", return_value=""),
            patch("classes.Tts.get_verbose", return_value=False),
        ):
            tts = TTS()
            with self.assertRaises(RuntimeError) as ctx:
                tts._resolve_qwen3_api_name()
            self.assertIn("qwen3_tts_reference_audio", str(ctx.exception))

    def test_explicit_do_job_with_reference_ok(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with (
                patch("classes.Tts.get_tts_backend", return_value="qwen3"),
                patch("classes.Tts.get_qwen3_tts_api_name", return_value="/do_job"),
                patch("classes.Tts.get_qwen3_tts_reference_audio", return_value=path),
                patch("classes.Tts.get_verbose", return_value=False),
            ):
                tts = TTS()
                self.assertEqual(tts._resolve_qwen3_api_name(), "/do_job")
        finally:
            os.unlink(path)


@unittest.skipUnless(os.environ.get("QWEN3_SMOKE") == "1", "set QWEN3_SMOKE=1 for live Gradio /do_job smoke test")
class Qwen3LiveSmokeTests(unittest.TestCase):
    def test_synthesize_short_text_via_do_job(self) -> None:
        import numpy as np
        import soundfile as sf
        from gradio_client.exceptions import AppError

        from config import ROOT_DIR, runtime_config_overrides

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        out_path = os.path.join(ROOT_DIR, ".mp", "smoke_qwen3_do_job.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            sf.write(wav_path, np.zeros(2400, dtype=np.float32), 24000)
            with runtime_config_overrides(
                {
                    "tts_backend": "qwen3",
                    "qwen3_tts_url": os.environ.get("QWEN3_TTS_URL", "http://127.0.0.1:7862"),
                    "qwen3_tts_api_name": "/do_job",
                    "qwen3_tts_reference_audio": wav_path,
                    "qwen3_tts_voices_dropdown": "使用参考音频",
                    "verbose": True,
                }
            ):
                try:
                    TTS().synthesize("测试。", out_path)
                except AppError as e:
                    self.skipTest(f"Gradio /do_job rejected request (use real reference clip or check server): {e}")
            self.assertTrue(os.path.isfile(out_path))
        finally:
            if os.path.isfile(wav_path):
                os.unlink(wav_path)


if __name__ == "__main__":
    unittest.main()
