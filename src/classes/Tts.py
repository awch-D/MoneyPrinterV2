import os
import re
import numpy as np
import soundfile as sf
from kittentts import KittenTTS as KittenModel

from config import ROOT_DIR, get_tts_voice

KITTEN_MODEL = "KittenML/kitten-tts-mini-0.8"
KITTEN_SAMPLE_RATE = 24000
# KittenTTS ONNX 模型对输入长度有限制，超过此字符数分句处理
MAX_CHUNK_CHARS = 100


def _split_sentences(text: str) -> list[str]:
    """按标点分句，每句不超过 MAX_CHUNK_CHARS 字符"""
    parts = re.split(r'(?<=[。！？.!?])', text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 超长句再按逗号切
        if len(part) > MAX_CHUNK_CHARS:
            sub = re.split(r'(?<=[，,、；;])', part)
            buf = ""
            for s in sub:
                if len(buf) + len(s) > MAX_CHUNK_CHARS and buf:
                    chunks.append(buf.strip())
                    buf = s
                else:
                    buf += s
            if buf.strip():
                chunks.append(buf.strip())
        else:
            chunks.append(part)
    return chunks or [text]


class TTS:
    def __init__(self) -> None:
        self._model = KittenModel(KITTEN_MODEL)
        self._voice = get_tts_voice()

    def synthesize(self, text, output_file=os.path.join(ROOT_DIR, ".mp", "audio.wav")):
        chunks = _split_sentences(text)
        if len(chunks) == 1:
            audio = self._model.generate(chunks[0], voice=self._voice)
        else:
            parts = []
            for chunk in chunks:
                parts.append(self._model.generate(chunk, voice=self._voice))
            audio = np.concatenate(parts)
        sf.write(output_file, audio, KITTEN_SAMPLE_RATE)
        return output_file
