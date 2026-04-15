import os
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import requests
import soundfile as sf

from config import (
    ROOT_DIR,
    get_qwen3_tts_api_name,
    get_qwen3_tts_batch,
    get_qwen3_tts_http_timeout_seconds,
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
    # MiniMax TTS configurations
    get_minimax_api_key,
    get_minimax_base_url,
    get_minimax_model,
    get_minimax_voice_id,
    get_minimax_reference_audio,
    get_minimax_prompt_text,
    get_minimax_speed,
    get_minimax_pitch,
    get_minimax_emotion,
    get_minimax_max_chunk_chars,
    get_minimax_http_timeout,
)
from status import info, warning

# Gradio qwen3-tts endpoints (see Client.view_api() on your app).
_DO_JOB_T_BASES = frozenset({"do_job_t"})
_DO_JOB_BASES = frozenset({"do_job"})
_do_job_prompt_hint_logged = False


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
    if isinstance(out, dict) and isinstance(out.get("__local_path"), str):
        p = str(out["__local_path"])
        return p if os.path.isfile(p) else None
    if isinstance(out, str) and os.path.isfile(out):
        return out
    return None


class TTS:
    def __init__(self) -> None:
        self._backend = get_tts_backend()
        if self._backend in ("qwen3_http", "qwen3_gradio", "qwen3"):
            self._backend = "qwen3"
            self._qwen3_client = None
        elif self._backend == "minimax":
            self._backend = "minimax"
            # Initialize MiniMax TTS client
            self._minimax_api_key = get_minimax_api_key()
            self._minimax_base_url = get_minimax_base_url()
            self._minimax_model = get_minimax_model()
            if not self._minimax_api_key:
                raise RuntimeError("MiniMax API key is required. Set 'minimax_api_key' in config.json")
        else:
            self._backend = "kitten"
            from kittentts import KittenTTS as KittenModel

            self._model = KittenModel(KITTEN_MODEL)
            self._voice = get_tts_voice()

    def _qwen3_client_singleton(self):
        if self._qwen3_client is None:
            import httpx
            from gradio_client import Client

            url = get_qwen3_tts_url()
            timeout_s = get_qwen3_tts_http_timeout_seconds()
            if timeout_s is None:
                self._qwen3_client = Client(url)
            else:
                # Gradio uses long-lived SSE; a single float caps *read* too and can break
                # predictions that go quiet on the wire during GPU work. Bound write/pool only.
                self._qwen3_client = Client(
                    url,
                    httpx_kwargs={
                        "timeout": httpx.Timeout(
                            connect=30.0,
                            read=None,
                            write=float(timeout_s),
                            pool=120.0,
                        )
                    },
                )
                if get_verbose():
                    info(
                        f"qwen3 gradio_client httpx write_timeout={int(timeout_s)}s "
                        "(read unlimited for SSE; omit qwen3_tts_http_timeout_seconds for Gradio defaults)"
                    )
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
                raise RuntimeError(
                    "qwen3_tts_api_name is /do_job but qwen3_tts_reference_audio is missing or not a file. "
                    "Set qwen3_tts_reference_audio in config.json to an absolute path to reference audio (WAV/MP3), "
                    "or use qwen3_tts_api_name /do_job_t or auto (auto picks /do_job only when the file exists)."
                )
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

    def _qwen3_do_job_via_rest(
        self,
        *,
        voices_dropdown: str,
        text: str,
        prompt_text: str,
        reference_audio_path: str,
        speed: float,
        chunk_size: int,
        batch: int,
        lang: str,
        model_type: str,
    ) -> dict:
        """
        Call qwen3-tts Gradio /do_job via REST to match UI behavior:
        1) POST /gradio_api/upload (reference audio)
        2) POST /gradio_api/call/do_job
        3) GET SSE stream /gradio_api/call/do_job/{event_id}
        4) Download resulting wav via returned FileData url
        Returns {"__local_path": "..."} for _normalize_gradio_output().
        """
        base = get_qwen3_tts_url().rstrip("/")

        # 1) Upload reference audio (UI does this implicitly)
        with open(reference_audio_path, "rb") as f:
            up = requests.post(
                f"{base}/gradio_api/upload",
                files={"files": (os.path.basename(reference_audio_path), f, "audio/wav")},
                timeout=30,
            )
        up.raise_for_status()
        server_path = up.json()[0]

        filedata = {"path": server_path, "meta": {"_type": "gradio.FileData"}}

        # If prompt_text is empty, mimic UI flow: auto-recognize prompt text from reference audio.
        # Many voice-clone pipelines rely on prompt_text matching the reference audio to stay stable.
        pt = (prompt_text or "").strip()
        if not pt:
            try:
                rec = requests.post(
                    f"{base}/gradio_api/call/prompt_wav_recognition",
                    json={"data": [filedata]},
                    timeout=30,
                )
                rec.raise_for_status()
                rec_eid = rec.json()["event_id"]
                rec_sse = requests.get(
                    f"{base}/gradio_api/call/prompt_wav_recognition/{rec_eid}",
                    stream=True,
                    timeout=(10, 300),
                )
                rec_sse.raise_for_status()
                rec_data_json: str | None = None
                for raw in rec_sse.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if raw.startswith("data: "):
                        rec_data_json = raw[len("data: ") :]
                if rec_data_json:
                    out = requests.models.complexjson.loads(rec_data_json)
                    if isinstance(out, str) and out.strip():
                        pt = out.strip()
            except Exception:
                # Best-effort: fall back to caller-provided prompt_text/instruct
                pt = (prompt_text or "").strip()

        payload = [
            voices_dropdown,
            text,
            pt,
            filedata,
            float(speed),
            int(chunk_size),
            int(batch),
            str(lang),
            str(model_type),
        ]

        call = requests.post(
            f"{base}/gradio_api/call/do_job",
            json={"data": payload},
            timeout=30,
        )
        call.raise_for_status()
        event_id = call.json()["event_id"]

        # 3) Stream SSE to completion
        sse = requests.get(
            f"{base}/gradio_api/call/do_job/{event_id}",
            stream=True,
            timeout=(10, 900),
        )
        sse.raise_for_status()

        data_json: str | None = None
        for raw in sse.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("event: complete"):
                continue
            if raw.startswith("data: "):
                candidate = raw[len("data: ") :].strip()
                # Gradio SSE may emit trailing "null" heartbeat/progress lines.
                # Keep the last non-null payload as the actual result.
                if candidate and candidate.lower() != "null":
                    data_json = candidate
        if not data_json:
            raise RuntimeError("Qwen3 TTS /do_job: missing SSE data payload")

        # Gradio returns [audio_filedata, download_filedata]
        out_list = requests.models.complexjson.loads(data_json)
        if not isinstance(out_list, list) or not out_list:
            raise RuntimeError(f"Qwen3 TTS /do_job: unexpected SSE data: {data_json[:200]!r}")
        audio_fd = out_list[0]
        if not isinstance(audio_fd, dict) or not isinstance(audio_fd.get("url"), str):
            raise RuntimeError(f"Qwen3 TTS /do_job: missing audio url in {audio_fd!r}")

        # 4) Download the wav
        wav_url = audio_fd["url"]
        r = requests.get(wav_url, timeout=600)
        r.raise_for_status()

        fd, tmp_path = tempfile.mkstemp(prefix="qwen3_tts_", suffix=".wav")
        os.close(fd)
        with open(tmp_path, "wb") as wf:
            wf.write(r.content)
        return {"__local_path": tmp_path}

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
            prompt_cfg = get_qwen3_tts_prompt_text()
            # Keep prompt_text empty to let /do_job auto-recognize transcript from reference audio.
            # Using instruct as fallback here often causes voice drift in clone mode.
            prompt = prompt_cfg
            if not prompt_cfg and get_verbose():
                warning(
                    "qwen3_tts_prompt_text 为空，/do_job 将自动识别参考音频文本；"
                    "若识别失败再手动填写与参考音频逐字一致的 prompt_text。"
                )
            # Gradio Audio expects uploaded FileData; raw path strings often fail on /do_job.
            # qwen3-tts Gradio Number(batch) rejects float 8.0 but accepts int 8.
            batch = int(get_qwen3_tts_batch())
            global _do_job_prompt_hint_logged
            if prompt_cfg and get_verbose() and not _do_job_prompt_hint_logged:
                _do_job_prompt_hint_logged = True
                info(
                    "/do_job: qwen3_tts_prompt_text must match reference audio verbatim; "
                    "unrelated text (e.g. after swapping reference file) can hang or confuse qwen3-tts."
                )
            return self._qwen3_do_job_via_rest(
                voices_dropdown=get_qwen3_tts_voices_dropdown(),
                text=text_chunk,
                prompt_text=prompt,
                reference_audio_path=ref,
                speed=get_qwen3_tts_speed(),
                chunk_size=get_qwen3_tts_gradio_chunk_size(),
                batch=batch,
                lang=get_qwen3_tts_lang(),
                model_type=get_qwen3_tts_model_type(),
            )

        args = self._build_qwen3_predict_args(text_chunk, instruct)
        return client.predict(*args, api_name=name)

    def _synthesize_qwen3_gradio(
        self,
        text: str,
        output_file: str,
        *,
        no_sentence_split: bool = False,
    ) -> str:
        client = self._qwen3_client_singleton()
        instruct = get_qwen3_tts_instruct()
        if no_sentence_split:
            t = text.strip()
            if not t:
                raise RuntimeError("Qwen3 TTS: empty text with no_sentence_split=True")
            chunks = [t]
        else:
            max_chars = max(50, get_qwen3_tts_max_chunk_chars())
            chunks = _split_sentences(text, max_chars)
        api_name = self._resolve_qwen3_api_name()

        wav_paths: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            info(
                f"Qwen3 TTS: Gradio {api_name} chunk {i}/{len(chunks)} "
                f"({len(chunk)} chars) - waiting for server..."
            )
            sys.stdout.flush()
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

    def _minimax_prosody(self, emotion: str) -> dict:
        """根据情绪返回韵律参数"""
        prosody_map = {
            "neutral": {"speed": 1.0, "pitch": 0},
            "happy": {"speed": 1.1, "pitch": 2},
            "sad": {"speed": 0.9, "pitch": -2},
            "angry": {"speed": 1.2, "pitch": 3},
            "calm": {"speed": 0.8, "pitch": -1},
        }
        return prosody_map.get(emotion, prosody_map["neutral"])

    def _minimax_emotion_mapping(self, emotion: str) -> str:
        """映射情绪到MiniMax支持的值"""
        emotion_map = {
            "neutral": "neutral",
            "happy": "happy", 
            "sad": "sad",
            "angry": "angry",
            "calm": "calm",
        }
        return emotion_map.get(emotion, "neutral")

    def _minimax_enrich_text(self, emotion: str, text: str) -> str:
        """根据情绪丰富文本"""
        if emotion == "happy":
            return f"[开心] {text}"
        elif emotion == "sad":
            return f"[难过] {text}"
        elif emotion == "angry":
            return f"[生气] {text}"
        elif emotion == "calm":
            return f"[平静] {text}"
        return text

    async def _synthesize_minimax_chunk(self, text_chunk: str) -> bytes:
        """使用MiniMax API合成单个文本块"""
        import httpx
        
        # 获取配置
        emotion = get_minimax_emotion()
        prosody = self._minimax_prosody(emotion)
        enriched_text = self._minimax_enrich_text(emotion, text_chunk)
        
        # 检查是否有参考音频进行声音克隆
        reference_audio = get_minimax_reference_audio()
        voice_id = get_minimax_voice_id()
        
        # 如果有参考音频，尝试使用声音克隆
        if reference_audio and os.path.isfile(reference_audio):
            try:
                from classes.MinimaxVoiceClone import MinimaxVoiceClone
                
                info(f"MiniMax TTS: 使用参考音频进行声音克隆: {reference_audio}")
                
                # 创建声音克隆客户端
                clone_client = MinimaxVoiceClone(self._minimax_api_key, self._minimax_base_url)
                
                # 获取prompt_text
                prompt_text = get_minimax_prompt_text()
                
                # 进行声音克隆并合成
                clone_result = await clone_client.clone_voice(
                    clone_audio_path=reference_audio,
                    prompt_text=prompt_text,
                    test_text=enriched_text,
                    model=self._minimax_model
                )
                
                # 使用克隆的声音合成
                return await clone_client.synthesize_with_cloned_voice(
                    text=enriched_text,
                    voice_id=clone_result["voice_id"],
                    model=self._minimax_model,
                    speed=get_minimax_speed() * prosody["speed"],
                    pitch=get_minimax_pitch() + prosody["pitch"],
                    emotion=self._minimax_emotion_mapping(emotion)
                )
                
            except Exception as e:
                warning(f"声音克隆失败，回退到默认音色: {e}")
                # 继续使用默认音色
        
        # 使用默认音色进行合成
        base_url = self._minimax_base_url
        if not base_url.endswith("/t2a_v2"):
            base_url = f"{base_url}/v1/t2a_v2"
        
        headers = {
            "Authorization": f"Bearer {self._minimax_api_key}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": self._minimax_model,
            "text": enriched_text,
            "stream": False,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": get_minimax_speed() * prosody["speed"],
                "vol": 1,
                "pitch": get_minimax_pitch() + prosody["pitch"],
                "emotion": self._minimax_emotion_mapping(emotion),
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "wav",
                "channel": 1,
            },
            "output_format": "hex",
            "subtitle_enable": False,
        }
        
        if get_verbose():
            info(
                f"MiniMax TTS: REQUEST model={body['model']}, text={repr(body['text'])[:100]}..., "
                f"voice={body['voice_setting']['voice_id']}, emotion={body['voice_setting']['emotion']}, "
                f"speed={body['voice_setting']['speed']}, pitch={body['voice_setting']['pitch']}"
            )
        
        timeout = get_minimax_http_timeout()
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(base_url, headers=headers, json=body)
            resp.raise_for_status()
            
            data = resp.json()
            base_resp = data.get("base_resp") or {}
            
            if base_resp.get("status_code") not in (None, 0):
                raise RuntimeError(
                    base_resp.get("status_msg") or f"MiniMax TTS error: {base_resp}"
                )
            
            audio_hex = (data.get("data") or {}).get("audio")
            if not audio_hex:
                raise RuntimeError(f"MiniMax TTS 返回空音频: {data}")
                
            return bytes.fromhex(audio_hex)

    def _synthesize_minimax(self, text: str, output_file: str) -> str:
        """使用MiniMax TTS合成语音"""
        import asyncio
        
        # 分割文本为块
        max_chars = get_minimax_max_chunk_chars()
        chunks = _split_sentences(text, max_chars)
        
        if get_verbose():
            info(f"MiniMax TTS: Processing {len(chunks)} chunks, max_chars={max_chars}")
        
        async def synthesize_all_chunks():
            audio_chunks = []
            for i, chunk in enumerate(chunks, 1):
                if get_verbose():
                    info(f"MiniMax TTS: Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
                
                audio_bytes = await self._synthesize_minimax_chunk(chunk)
                audio_chunks.append(audio_bytes)
            
            return audio_chunks
        
        # 运行异步合成
        try:
            # 检查是否已经在事件循环中
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，创建新的线程来运行
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(synthesize_all_chunks())
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                audio_chunks = future.result()
                
        except RuntimeError:
            # 没有运行的事件循环，直接运行
            audio_chunks = asyncio.run(synthesize_all_chunks())
        
        if len(audio_chunks) == 1:
            # 单个块，直接保存
            with open(output_file, "wb") as f:
                f.write(audio_chunks[0])
        else:
            # 多个块，需要合并
            arrays = []
            sr = 32000
            
            for audio_bytes in audio_chunks:
                # 将字节数据写入临时文件，然后读取
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    data, file_sr = sf.read(tmp_path)
                    sr = file_sr
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    arrays.append(np.asarray(data, dtype=np.float32))
                finally:
                    os.unlink(tmp_path)
            
            # 合并所有音频块
            merged = np.concatenate(arrays)
            sf.write(output_file, merged, sr)
        
        if get_verbose():
            info(f"MiniMax TTS: Audio saved to {output_file}")
        
        return output_file

    def synthesize(
        self,
        text,
        output_file=os.path.join(ROOT_DIR, ".mp", "audio.wav"),
        *,
        qwen3_no_sentence_split: bool = False,
    ):
        if self._backend == "qwen3":
            return self._synthesize_qwen3_gradio(
                text,
                output_file,
                no_sentence_split=qwen3_no_sentence_split,
            )
        elif self._backend == "minimax":
            return self._synthesize_minimax(text, output_file)

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
