# Novel chapter capability (一章一集)

The `novel_chapter` capability turns **one plain-text chapter file** into a single narrated video: structured scenes from the LLM, **one generated image per scene**, **per-scene TTS** so picture duration matches that beat, then the same **subtitles + BGM + encode** path as the short pipeline.

## CLI

From the project root (with `config.json` filled for script API, image API, and TTS):

```bash
python src/main.py --capability novel_chapter --chapter-file path/to/chapter.txt --language Chinese --orientation landscape
```

- `--orientation` (`landscape` default, or `portrait`) applies **only for this run** via in-memory config overrides: it sets both final video aspect (`video_output_aspect`) and image generation size (`nanobanana2_aspect_ratio`) to `16:9` or `9:16` so framing stays consistent.
- `--topic` optional label used in logs / JSON output (defaults to the chapter filename).
- `--niche` is unused for this capability.

## How it works

1. **Chapter analysis** (`src/novel/chapter_analyzer.py`): OpenAI-compatible `script_api_*` returns JSON with `style_bible`, `characters[]` (stable `look` strings), and `segments[]` (`narration`, `scene_summary`, `image_prompt`, `visible_character_ids`).
2. **Consistency**: Each image request prepends the style bible and visible characters’ looks to the segment `image_prompt` (`build_merged_image_prompt`), then appends optional `novel_chapter_image_prompt_suffix`, then the optional **global画风** from `image_prompt_style` / `image_prompt_style_preset` (see `Configuration.md`).
3. **Audio timeline** (`src/novel/chapter_audio.py`, `novel_audio_pipeline` in `config.json`):
   - **`segment_merge` (default):** each `narration` → its own WAV → measured durations → merge with optional **crossfade** (`audio_merge_crossfade_ms`).
   - **`full_track_whisperx`:** one TTS call on the concatenation of cleaned lines (single timbre), then **WhisperX forced alignment** (`src/novel/whisperx_segment_align.py`) on the whole WAV to recover **per-segment durations** for picture + script subtitles. Requires `pip install whisperx` (PyTorch). Use `whisperx_device` / `whisperx_language_code` in config.
4. **Video** (`ShortVideoPipeline.combine_timeline`): One still per segment, `duration ==` that segment’s length (scaled if needed to match the merged WAV). **Ken Burns:** `video_ken_burns_dynamic_zoom` (default `false`) keeps **fixed zoom** (`zoom_max` follows `zoom_min`) and only **horizontal pan** moves (`video_ken_burns_pan_extent`), reducing zoom-driven jitter. **Subtitles:** same cleaned text as TTS and per-segment durations—one SRT cue per segment—no Whisper on the narration for novel runs with `subtitle_segment_texts`.
5. **Manifest** (`.mp/last_timeline_manifest.json`): `subtitle_segment_texts`, `novel_audio_pipeline`, and paths for recombine.

## Configuration

- Storyboard segmentation targets **shot-level** units (one beat per segment, narration targeting **5–8s** VO; for Chinese ~3–4 chars/s, usually one short sentence; do not use one segment per novel paragraph if that exceeds ~8s; explicit shot scale in each `image_prompt`, ultra-wide→tight progression where tension builds). See `src/novel/chapter_analyzer.py`.
- Scene count is **not** capped in code: long chapters yield more segments. For a **temporary** cap when experimenting, use `python scripts/preview_chapter_segments.py --max-segments N`.
- `novel_chapter_image_prompt_suffix` (optional): appended to every merged image prompt for novel chapters (e.g. shared avoid / quality line); see `Configuration.md`.
- All existing keys for `script_api_*`, `nanobanana2_*`, TTS, Whisper/AssemblyAI, fonts, and threads apply unchanged.
- `image_prompt_style` / `image_prompt_style_preset`: shared with the short pipeline; appended to every merged image prompt before the image API call.

## Extending capabilities

Other packaged flows (e.g. future `ppt_narrative`) should register in `src/capabilities/registry.py`, reuse `combine_timeline` when “one visual per spoken segment” applies, and keep CLI-facing options in `src/main.py`.
