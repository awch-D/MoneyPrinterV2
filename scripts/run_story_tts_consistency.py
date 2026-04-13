#!/usr/bin/env python3
"""
Read a plain-text story file and synthesize merged WAV using the same TTS stack as novel_chapter.

**参考音频（Qwen3 /do_job）** 完全由 ``config.json`` 决定：须设置有效的 ``qwen3_tts_reference_audio``、
``qwen3_tts_voices_dropdown`` 一般为 ``使用参考音频``、``qwen3_tts_prompt_text`` 为参考 wav 的转写；
``qwen3_tts_api_name`` 为 ``auto`` 或 ``/do_job``。脚本启动时会打印是否走 ``/do_job``（只有该路径才会把参考文件交给 Gradio）。可加 ``--require-do-job`` 强制校验。

- ``segment`` (default): one ``TTS()`` instance, one ``synthesize`` per paragraph, then merge
  (same pattern as ``novel_audio_pipeline: segment_merge``).
- ``full``: concatenate cleaned lines and call ``synthesize`` once with
  ``qwen3_no_sentence_split=True`` (same as ``full_track_whisperx``): **one** Qwen3 predict
  for the whole text, no client-side splitting on ``。！？``.

**音色一致性（粗测）**：``--report-mel`` 使用对数 Mel 谱沿时间取均值作为嵌入，计算各段与
``qwen3_tts_reference_audio`` 的余弦相似度，以及段与段之间的平均相似度（仅 ``segment`` 模式）。
此为声学纹理近似指标，非专业声纹模型；数值越高通常表示与参考/彼此越接近。

Paragraphs are separated by blank lines; if none, non-empty lines are used as segments.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from classes.Tts import TTS  # noqa: E402
from config import (  # noqa: E402
    ROOT_DIR,
    get_qwen3_tts_api_name,
    get_qwen3_tts_prompt_text,
    get_qwen3_tts_reference_audio,
    get_qwen3_tts_voices_dropdown,
    get_tts_backend,
)
from novel.chapter_audio import (  # noqa: E402
    clean_narration_for_tts,
    synthesize_full_track_to_wav,
    synthesize_segments_to_merged_wav,
)


def _segments_from_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError(f"empty file: {path}")
    parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if len(parts) <= 1 and "\n\n" not in raw:
        parts = [line.strip() for line in raw.split("\n") if line.strip()]
    return parts


def _print_qwen3_reference_banner(tts: TTS) -> str | None:
    """Print how reference audio will be used; return resolved Gradio API name for qwen3, else None."""
    backend = get_tts_backend().strip().lower()
    if backend not in ("qwen3", "qwen3_http", "qwen3_gradio"):
        print(f"TTS backend={backend!r}（非 qwen3 Gradio，不使用 qwen3_tts_reference_audio）")
        return None
    ref = get_qwen3_tts_reference_audio()
    ref_ok = bool(ref and os.path.isfile(ref))
    print("--- Qwen3 参考音 ---")
    print("  reference_audio:", ref if ref_ok else ref, "→", "文件存在" if ref_ok else "缺失")
    print("  voices_dropdown:", repr(get_qwen3_tts_voices_dropdown()))
    print("  qwen3_tts_api_name:", repr(get_qwen3_tts_api_name()))
    pt = get_qwen3_tts_prompt_text()
    print("  prompt_text 字符数:", len(pt), "(须与参考 wav 内容一致)")
    api = tts._resolve_qwen3_api_name()
    print("  将调用的 Gradio API:", repr(api))
    if api != "/do_job":
        print(
            " 警告: 未走 /do_job，参考 wav 不会参与合成；请配置参考文件并设 api为 auto 或 /do_job。",
            file=sys.stderr,
        )
    else:
        print("  将使用参考音频链路 (/do_job)。")
    print("---")
    return api


def _mel_log_mean_embedding(
    wav_path: str,
    *,
    target_sr: int = 16_000,
    max_ref_seconds: float = 12.0,
) -> "torch.Tensor":
    """Mean log-mel vector (n_mels,) for cosine similarity; reference uses center crop if long."""
    import torch
    import torch.nn.functional as F
    import torchaudio

    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    n = wav.shape[-1]
    max_samples = int(max_ref_seconds * target_sr)
    if n > max_samples:
        start = max(0, (n - max_samples) // 2)
        wav = wav[..., start : start + max_samples]

    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        f_min=40,
        f_max=7600,
    )
    mel = mel_t(wav)
    mel = torch.log(mel.clamp_min(1e-6))
    emb = mel.mean(dim=-1).squeeze(0)
    emb = F.normalize(emb, dim=0)
    return emb


def _print_mel_voice_report(
    reference_wav: str,
    segment_wav_paths: list[str],
) -> None:
    import torch

    if not reference_wav or not os.path.isfile(reference_wav):
        print("--- Mel 音色报告 ---\n  跳过：参考 wav 不存在\n---")
        return
    try:
        ref_e = _mel_log_mean_embedding(reference_wav)
    except Exception as exc:  # noqa: BLE001
        print("--- Mel 音色报告 ---\n  参考音频嵌入失败:", exc, "\n---")
        return

    print("--- Mel 音色报告（对数 Mel 均值 vs 参考；余弦相似度 [-1,1]，越大越接近）---")
    seg_es: list[torch.Tensor] = []
    for i, p in enumerate(segment_wav_paths, 1):
        try:
            e = _mel_log_mean_embedding(p, max_ref_seconds=3600.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  段 {i}: 嵌入失败 {exc}")
            continue
        seg_es.append(e)
        sim = float(torch.dot(ref_e, e).item())
        print(f"  段 {i} vs 参考: {sim:.4f}  ({os.path.basename(p)})")

    if len(seg_es) >= 2:
        pair_sims: list[float] = []
        for a in range(len(seg_es)):
            for b in range(a + 1, len(seg_es)):
                pair_sims.append(float(torch.dot(seg_es[a], seg_es[b]).item()))
        mean_p = sum(pair_sims) / len(pair_sims)
        print(
            f"  段间两两平均相似度: {mean_p:.4f}（共 {len(pair_sims)} 对；"
            f"min {min(pair_sims):.4f}, max {max(pair_sims):.4f}）",
        )
    print("---")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("story_file", help="Path to .txt (e.g. stories/tianbao_short.txt)")
    ap.add_argument(
        "-o",
        "--output",
        default="",
        help="Merged WAV path (default: .mp/story_tts_<name>.wav)",
    )
    ap.add_argument(
        "--mode",
        choices=("segment", "full"),
        default="segment",
        help="segment = per-paragraph TTS + merge; full = one TTS on joined text",
    )
    ap.add_argument(
        "--require-do-job",
        action="store_true",
        help="Exit with error if Qwen3 resolved API is not /do_job (ensures reference-audio path)",
    )
    ap.add_argument(
        "--report-mel",
        action="store_true",
        help="After synthesis, print log-mel cosine similarity vs reference and between segments",
    )
    ap.add_argument(
        "--keep-segments",
        default="",
        help="If set (segment mode), copy per-segment WAVs into this directory as seg_01.wav, ...",
    )
    args = ap.parse_args()

    path = os.path.abspath(args.story_file)
    if not os.path.isfile(path):
        print("ERROR: not a file:", path, file=sys.stderr)
        return 1

    segments = _segments_from_file(path)
    print(f"segments: {len(segments)} from {path}")
    for i, s in enumerate(segments, 1):
        preview = s.replace("\n", " ")[:60] + ("…" if len(s) > 60 else "")
        print(f"  {i}. {preview}")

    tts = TTS()
    resolved = _print_qwen3_reference_banner(tts)
    if args.require_do_job and resolved is not None and resolved != "/do_job":
        print("ERROR: --require-do-job but resolved API is not /do_job", file=sys.stderr)
        return 1
    segment_paths_for_report: list[str] = []
    if args.mode == "full":
        lines = [clean_narration_for_tts(s).strip() for s in segments]
        if any(not x for x in lines):
            print("ERROR: empty line after clean", file=sys.stderr)
            return 1
        merged = synthesize_full_track_to_wav(lines, tts)
        if args.report_mel:
            segment_paths_for_report = [merged]
    else:
        _segs, durs, merged = synthesize_segments_to_merged_wav(segments, tts)
        print("segment durations (s):", [round(d, 2) for d in durs])
        segment_paths_for_report = list(_segs)
        keep = args.keep_segments.strip()
        if keep:
            kd = os.path.abspath(keep)
            os.makedirs(kd, exist_ok=True)
            for idx, sp in enumerate(_segs, 1):
                dest = os.path.join(kd, f"seg_{idx:02d}.wav")
                shutil.copy2(sp, dest)
            print("segment WAV copies:", kd)

    out = args.output.strip()
    if not out:
        base = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(ROOT_DIR, ".mp", f"story_tts_{base}.wav")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if os.path.abspath(merged) != os.path.abspath(out):
        shutil.copy2(merged, out)
    else:
        out = merged

    print("OK", out, os.path.getsize(out), "bytes")
    if args.report_mel:
        ref = get_qwen3_tts_reference_audio()
        _print_mel_voice_report(ref, segment_paths_for_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
