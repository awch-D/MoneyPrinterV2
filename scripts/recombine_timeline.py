#!/usr/bin/env python3
"""
Re-compose video from a timeline manifest (images + merged narration + per-segment durations).

No script API, image API, or TTS — only MoviePy + Whisper + BGM like a normal combine_timeline run.

After each successful `novel_chapter` run, `.mp/last_timeline_manifest.json` is written automatically.

Example:

  PYTHONPATH=src python scripts/recombine_timeline.py --manifest .mp/last_timeline_manifest.json

You can also build a manifest by hand:

  {
    "merged_wav": "/abs/path/to/novel_merged_xxx.wav",
    "segment_durations": [1.2, 2.0, ...],
    "image_paths": ["/abs/path/to/seg1.jpg", ...]
  }

If `segment_durations` is omitted, provide `segment_wavs` (same list) — durations are measured with ffprobe/MoviePy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from moviepy import AudioFileClip  # noqa: E402

from pipeline.short_video_pipeline import ShortVideoPipeline  # noqa: E402
from providers.image_api_provider import ImageApiProvider  # noqa: E402
from providers.script_api_provider import ScriptApiProvider  # noqa: E402
from status import info, success  # noqa: E402


def _wav_duration(path: str) -> float:
    c = AudioFileClip(path)
    try:
        return float(c.duration or 0.0)
    finally:
        c.close()


def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_paths(d: dict) -> tuple[list[str], list[float], str]:
    merged = (d.get("merged_wav") or "").strip()
    if not merged:
        raise ValueError("manifest: missing merged_wav")
    merged = os.path.abspath(merged)

    images = d.get("image_paths") or []
    if not isinstance(images, list) or not images:
        raise ValueError("manifest: missing or empty image_paths")
    images = [os.path.abspath(str(p)) for p in images]

    durs = d.get("segment_durations")
    if durs is not None:
        if not isinstance(durs, list) or len(durs) != len(images):
            raise ValueError("segment_durations must match image_paths length")
        return images, [float(x) for x in durs], merged

    seg_wavs = d.get("segment_wavs") or []
    if not isinstance(seg_wavs, list) or len(seg_wavs) != len(images):
        raise ValueError("Provide segment_durations or segment_wavs (same length as image_paths)")
    measured = [_wav_duration(os.path.abspath(str(p))) for p in seg_wavs]
    return images, measured, merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-compose video from timeline manifest JSON.")
    parser.add_argument(
        "--manifest",
        default=os.path.join(ROOT_DIR, ".mp", "last_timeline_manifest.json"),
        help="Path to JSON (default: .mp/last_timeline_manifest.json)",
    )
    args = parser.parse_args()

    mp = os.path.abspath(args.manifest)
    print(f"[recombine] 清单: {mp}", flush=True)
    if not os.path.isfile(mp):
        print(f"Manifest not found: {mp}", file=sys.stderr)
        return 1

    data = load_manifest(mp)
    print("[recombine] 解析清单并校验文件…", flush=True)
    if data.get("segment_durations") is None and data.get("segment_wavs"):
        n = len(data["segment_wavs"])
        print(f"[recombine] 从 {n} 个分段 WAV 读取时长（通常很快）…", flush=True)
    image_paths, durations, merged_wav = _resolve_paths(data)

    for p in image_paths + [merged_wav]:
        if not os.path.isfile(p):
            print(f"Missing file: {p}", file=sys.stderr)
            return 1

    topic = (data.get("topic") or "recombine").strip()
    info(f'Recombining timeline: "{topic}" ({len(image_paths)} segments)')
    print(
        "[recombine] 开始合成（Whisper 转写 + MoviePy 逐帧编码，常见 5～15 分钟；"
        "进度条会刷新在同一行，请耐心等待）…",
        flush=True,
    )

    pipeline = ShortVideoPipeline(ScriptApiProvider(), ImageApiProvider())
    video_path, subtitle_path = pipeline.combine_timeline(image_paths, durations, merged_wav)

    success(f"Wrote video: {video_path}")
    if subtitle_path:
        info(f"Subtitles: {subtitle_path}")
    print(f"[recombine] 完成: {video_path}", flush=True)
    print(video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
