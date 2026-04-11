import os
import re
import shutil
from typing import Any

import numpy as np
import soundfile as sf

from config import (
    ROOT_DIR,
    get_qwen3_tts_api_name,
    get_qwen3_tts_batch,
    get_qwen3_tts_gradio_chunk_size,
    get_qwen3_tts_instruct,
    get_qwen3_tts_lang,
    get_qwen3_tts_max_chunk_chars,
    get_qwen3_tts_model_type,
    get_qwen3_tts_predict_arg_order,
    get_qwen3_tts_prompt_text,
    get_qwen3_tts_reference_audio,
    get_qwen3_tts_speed,
    get_qwen3_tts_static_args,
    get_qwen3_tts_url,
    get_qwen3_tts_voices_dropdown,
    get_tts_backend,
    get_tts_voice,
    get_verbose,
)
from status import info, warning

# Gradio qwen3-tts endpoints (see Client.view_api() on your app).
_DO_JOB_T_BASES = frozenset({"do_job_t"})
_DO_JOB_BASES = frozenset({"do_job"})


def _normalize_api_name(api_name: str) -> str:
    name = api_name.strip()
    if not name:
        return "/do_job_t"
    if not name.startswith("/"):
        name = "/" + name
    return name


def _api_base(api_name: str) -> str:
    return _normalize_api_name(api_name).rsplit("/", 1)[-1]

# KittenTTS (local)
KITTEN_MODEL = "KittenML/kitten-tts-mini-0.8"
KITTEN_SAMPLE_RATE = 24000
MAX_CHUNK_CHARS_KITTEN = 100


def _split_sentences(text: str, max_chars: int) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])", text)
    chunks: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            sub = re.split(r"(?<=[，,、；;])", part)
            buf = ""
            for s in sub:
                if len(buf) + len(s) > max_chars and buf:
                    chunks.append(buf.strip())
                    buf = s
                else:
                    buf += s
            if buf.strip():
                chunks.append(buf.strip())
        else:
            chunks.append(part)
    return chunks or [text]


def _normalize_gradio_output(out: Any) -> str | None:
    if isinstance(out, (list, tuple)) and out:
        out = out[0]
    if isinstance(out, str) and os.path.isfile(out):
        return out
    return None


class TTS:
    def __init__(self) -> None:
        self._backend = get_tts_backend()
        if self._backend in ("qwen3_http", "qwen3_gradio", "qwen3"):
            self._backend = "qwen3"
            self._qwen3_client = None
        else:
            self._backend = "kitten"
            from kittentts import KittenTTS as KittenModel

            self._model = KittenModel(KITTEN_MODEL)
            self._voice = get_tts_voice()

    def _qwen3_client_singleton(self):
        if self._qwen3_client is None:
            from gradio_client import Client

            self._qwen3_client = Client(get_qwen3_tts_url())
        return self._qwen3_client

    def _resolve_qwen3_api_name(self) -> str:
        raw = get_qwen3_tts_api_name().strip()
        if raw.lower() == "auto":
            ref = get_qwen3_tts_reference_audio()
            chosen = "/do_job" if ref and os.path.isfile(ref) else "/do_job_t"
            if get_verbose():
                info(
                    f'qwen3_tts_api_name=auto -> using {chosen} '
                    f'({"reference_audio set" if chosen == "/do_job" else "no reference_audio"}).'
                )
            return chosen
        name = _normalize_api_name(raw)
        if _api_base(name) in _DO_JOB_BASES:
            ref = get_qwen3_tts_reference_audio()
            if not ref or not os.path.isfile(ref):
                if get_verbose():
                    warning(
                        "qwen3_tts_api_name is /do_job but qwen3_tts_reference_audio is missing or not a file; "
                        "falling back to /do_job_t."
                    )
                return "/do_job_t"
        return name

    def _build_qwen3_predict_args(self, chunk: str, instruct: str) -> list[Any]:
        order = get_qwen3_tts_predict_arg_order()
        dynamic: list[Any] = []
        for key in order:
            k = key.lower()
            if k in ("text", "chunk", "content"):
                dynamic.append(chunk)
            elif k in ("instruct", "instruction", "style", "prompt"):
                dynamic.append(instruct)
            else:
                raise ValueError(
                    f"Unknown qwen3_tts_predict_arg_order entry: {key!r} "
                    '(use "instruct" and "text")'
                )
        return list(get_qwen3_tts_static_args()) + dynamic

    def _predict_qwen3(self, client: Any, text_chunk: str, instruct: str, api_name: str) -> Any:
        name = _normalize_api_name(api_name)
        base = name.rsplit("/", 1)[-1]

        if base in _DO_JOB_T_BASES:
            return client.predict(
                text_chunk,
                instruct,
                get_qwen3_tts_gradio_chunk_size(),
                api_name=name,
            )

        if base in _DO_JOB_BASES:
            ref = get_qwen3_tts_reference_audio()
            prompt = get_qwen3_tts_prompt_text() or instruct
            return client.predict(
                get_qwen3_tts_voices_dropdown(),
                text_chunk,
                prompt,
                ref,
                get_qwen3_tts_speed(),
                get_qwen3_tts_gradio_chunk_size(),
                get_qwen3_tts_batch(),
                get_qwen3_tts_lang(),
                get_qwen3_tts_model_type(),
                api_name=name,
            )

        args = self._build_qwen3_predict_args(text_chunk, instruct)
        return client.predict(*args, api_name=name)

    def _synthesize_qwen3_gradio(self, text: str, output_file: str) -> str:
        client = self._qwen3_client_singleton()
        instruct = get_qwen3_tts_instruct()
        max_chars = max(50, get_qwen3_tts_max_chunk_chars())
        chunks = _split_sentences(text, max_chars)
        api_name = self._resolve_qwen3_api_name()

        wav_paths: list[str] = []
        for chunk in chunks:
            raw = self._predict_qwen3(client, chunk, instruct, api_name)
            path = _normalize_gradio_output(raw)
            if not path:
                raise RuntimeError(
                    f"Qwen3 TTS Gradio returned unexpected output: {type(raw).__name__} {raw!r}"
                )
            wav_paths.append(path)

        if len(wav_paths) == 1:
            shutil.copy2(wav_paths[0], output_file)
            return output_file

        arrays: list[np.ndarray] = []
        sr = 24000
        for path in wav_paths:
            data, file_sr = sf.read(path)
            sr = file_sr
            if data.ndim > 1:
                data = data.mean(axis=1)
            arrays.append(np.asarray(data, dtype=np.float32))
        merged = np.concatenate(arrays)
        sf.write(output_file, merged, sr)
        return output_file

    def synthesize(self, text, output_file=os.path.join(ROOT_DIR, ".mp", "audio.wav")):
        if self._backend == "qwen3":
            return self._synthesize_qwen3_gradio(text, output_file)

        chunks = _split_sentences(text, MAX_CHUNK_CHARS_KITTEN)
        if len(chunks) == 1:
            audio = self._model.generate(chunks[0], voice=self._voice)
        else:
            parts = []
            for chunk in chunks:
                parts.append(self._model.generate(chunk, voice=self._voice))
            audio = np.concatenate(parts)
        sf.write(output_file, audio, KITTEN_SAMPLE_RATE)
        return output_file
