import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from audio_merge import merge_wav_files


class AudioMergeTests(unittest.TestCase):
    def test_merge_two_wavs_same_sr(self) -> None:
        sr = 16000
        a = np.zeros(800, dtype=np.float32)
        b = np.ones(400, dtype=np.float32) * 0.25
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "a.wav")
            p2 = os.path.join(d, "b.wav")
            out = os.path.join(d, "out.wav")
            sf.write(p1, a, sr)
            sf.write(p2, b, sr)
            merge_wav_files([p1, p2], out)
            merged, m_sr = sf.read(out)
            self.assertEqual(m_sr, sr)
            self.assertEqual(len(merged), 1200)

    def test_merge_two_wavs_with_crossfade(self) -> None:
        sr = 16000
        fade_ms = 15.0
        fade_n = int(sr * fade_ms / 1000.0)  # 240
        a = np.zeros(800, dtype=np.float32)
        b = np.ones(400, dtype=np.float32) * 0.25
        n = min(fade_n, 800 // 2 - 1, 400 // 2 - 1)
        expect_len = 800 + 400 - n
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "a.wav")
            p2 = os.path.join(d, "b.wav")
            out = os.path.join(d, "out.wav")
            sf.write(p1, a, sr)
            sf.write(p2, b, sr)
            merge_wav_files([p1, p2], out, crossfade_ms=fade_ms)
            merged, m_sr = sf.read(out)
            self.assertEqual(m_sr, sr)
            self.assertEqual(len(merged), expect_len)


if __name__ == "__main__":
    unittest.main()
