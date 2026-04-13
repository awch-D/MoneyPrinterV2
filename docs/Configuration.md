# Configuration

All your configurations will be in a file in the root directory, called `config.json`, which is a copy of `config.example.json`. You can change the values in `config.json` to your liking.

For how to run the CLI (short vs novel chapter, orientation, examples), see [Usage.md](./Usage.md).

## CLI (capabilities and aspect)

`python src/main.py` supports:

- `--capability short` (default): existing topic/script short pipeline (`--niche` or `--script-file`).
- `--capability novel_chapter`: one chapter file → storyboard JSON → images + per-segment TTS timeline; requires `--chapter-file`.
- `--orientation landscape` (default) or `portrait`: temporary merge into config reads for that run only (does not rewrite `config.json`). See [NovelChapter.md](./NovelChapter.md).

## Values

- `verbose`: `boolean` - If `true`, the application will print out more information.
- `firefox_profile`: `string` - The path to your Firefox profile. This is used to use your Social Media Accounts without having to log in every time you run the application.
- `headless`: `boolean` - If `true`, the application will run in headless mode. This means that the browser will not be visible.
- `script_api_base_url`: `string` - OpenAI-compatible API base for script / storyboard chat (default: `https://api.openai.com/v1`). Trailing slash is stripped in code.
- `script_api_key`: `string` - Bearer token for that API. If empty, the app uses environment variable `SCRIPT_API_KEY`.
- `script_api_model`: `string` - Chat model id for completions (e.g. `gpt-4.1-mini`).
- `twitter_language`: `string` - The language that will be used to generate & post tweets.
- `nanobanana2_api_base_url`: `string` - Nano Banana 2 API base URL (default: `https://generativelanguage.googleapis.com/v1beta`).
- `nanobanana2_api_key`: `string` - API key for Nano Banana 2 (Gemini image API). If empty, MPV2 falls back to environment variable `GEMINI_API_KEY`.
- `nanobanana2_model`: `string` - Nano Banana 2 model name (default in code: `gemini-3.1-flash-image`; 网关若要求带后缀如 `…-preview-4k` 则按服务商文档填写).
- `nanobanana2_aspect_ratio`: `string` - **Ratio string** for the OpenAI-compatible image API (same contract as Gemini proxy / [Novel2Toon gemini-image-gen skill](../Novel2Toon/gemini-image-gen_副本/SKILL.md)): `1:1`, `16:9`, `9:16`, etc. Do **not** use pixel sizes like `1920x1080` (will be coerced or rejected by the proxy). Default in `config.example.json` is `9:16`. For a single run, `python src/main.py --orientation landscape|portrait` overrides this together with `video_output_aspect` so image size matches the final frame.
- `nanobanana2_image_timeout_seconds` / `nanobanana2_image_max_retries`: slow proxies (e.g. 4K) may need a higher timeout; see `config.example.json`.
- `image_prompt_style`: `string` - Optional **full** style block appended to every image prompt (short + novel chapter). If non-empty, it **overrides** `image_prompt_style_preset`. Use for fully custom styles without editing code.
- `image_prompt_style_preset`: `string` - When `image_prompt_style` is empty, selects a built-in template from `src/novel/image_style_presets.py`: `none` (no extra block), `han_guofeng_woodcut` (汉代国风木刻暗黑风，项目默认示例), `cinematic_clean` (英文电影干净风). Unknown keys log a warning and apply no style.
- `video_output_aspect`: `string` - Final MP4 layout: `16:9` (default landscape) or `9:16` / `portrait` for vertical HD (see `get_video_output_size()` in `src/config.py`).
- `video_fps`: `number` - Output frame rate for still sequences (default `30`).
- `video_ken_burns_enabled`: `boolean` - If `true`, each image uses slow zoom between `video_ken_burns_zoom_min` and `video_ken_burns_zoom_max` (code defaults `1.03`–`1.07`). Optional horizontal pan is controlled by `video_ken_burns_pan_extent` (default `0` = zoom only, less “shaky” feel than full pan).
- `video_ken_burns_pan_extent` / `video_ken_burns_pan_max_width_ratio`: pan strength and pixel cap; see `src/config.py`.
- `video_transition`: `string` - `none` | `page_flip` (between every segment) | `random_page_flip` (probabilistic). Alias: top-level `transition` is read if `video_transition` is empty. Example config uses `none` to avoid slide transitions that read as jitter.
- `video_page_flip_probability`: `number` - Used when `video_transition` is `random_page_flip` (default `0.35`).
- `video_page_flip_duration_seconds`: `number` - Length of each page-flip clip; adjacent segments are each shortened by half of this so total duration matches audio (default `0.38`).
- `video_transition_random_seed`: `number` | `null` - Fix random flips for reproducibility; `null` = nondeterministic.
- `audio_merge_crossfade_ms`: `number` - When merging per-segment novel chapter WAVs, linear crossfade length in milliseconds at each join (default `15`, max `80`; set `0` to disable). Shortens the merged file slightly; `combine_timeline` scales segment lengths to the merged WAV.
- `video_ken_burns_dynamic_zoom`: `boolean` - When `false` (default), Ken Burns uses **fixed** zoom (`zoom_max` is ignored for motion; only `zoom_min` applies) and **pan** only (`video_ken_burns_pan_extent`), reducing time-varying resize jitter.
- `novel_audio_pipeline`: `string` - `segment_merge` (default) or `full_track_whisperx` (one-shot TTS + WhisperX alignment; see `docs/NovelChapter.md`).
- `whisperx_device`: `string` - e.g. `cuda` or `cpu` for WhisperX alignment.
- `whisperx_language_code`: `string` - Passed to `whisperx.load_align_model` (e.g. `zh`, `en`).
- `novel_chapter_image_prompt_suffix`: `string` - Appended to every **merged** novel-chapter image prompt after `style_bible`, character looks, scene context, and segment `image_prompt` (before the global `image_prompt_style` / preset). Use for a shared “avoid / negative” line in natural language. Empty string disables.
- `threads`: `number` - The amount of threads that will be used to execute operations, e.g. writing to a file using MoviePy.
- `is_for_kids`: `boolean` - If `true`, the application will upload the video to YouTube Shorts as a video for kids.
- `google_maps_scraper`: `string` - The URL to the Google Maps scraper. This will be used to scrape Google Maps for local businesses. It is recommended to use the default value.
- `zip_url`: `string` - The URL to the ZIP file that contains the to be used Songs for the YouTube Shorts Automater.
- `email`: `object`:
    - `smtp_server`: `string` - Your SMTP server.
    - `smtp_port`: `number` - The port of your SMTP server.
    - `username`: `string` - Your email address.
    - `password`: `string` - Your email password.
- `google_maps_scraper_niche`: `string` - The niche you want to scrape Google Maps for.
- `scraper_timeout`: `number` - The timeout for the Google Maps scraper.
- `outreach_message_subject`: `string` - The subject of your outreach message. `{{COMPANY_NAME}}` will be replaced with the company name.
- `outreach_message_body_file`: `string` - The file that contains the body of your outreach message, should be HTML. `{{COMPANY_NAME}}` will be replaced with the company name.
- `stt_provider`: `string` - Provider for subtitle transcription. Default is `local_whisper`. Options:
    * `local_whisper` — runs the OpenAI **`whisper` CLI** (not faster-whisper).
    * `third_party_assemblyai`
- `whisper_cli_path`: `string` - Optional full path to the `whisper` executable (e.g. `/opt/homebrew/bin/whisper`). If empty, `PATH` is searched.
- `whisper_cli_timeout_seconds`: `number` - Subprocess timeout for the Whisper CLI (default `7200`).
- `whisper_model`: `string` - Passed to `whisper --model` (e.g. `base`, `small`, `medium`, `large-v3`, `turbo`). Legacy value `large-v3-turbo` is mapped to `turbo` for the CLI.
- `whisper_model_path`: `string` - If set, overrides `whisper_model` and is passed as `--model` (local checkpoint path supported by openai-whisper).
- `whisper_device`: `string` - Maps to `whisper --device` when not `auto` (`cpu`, `cuda`, etc.).
- `whisper_compute_type`: `string` - Maps to `whisper --fp16`: `int8` / `int8_*` → `--fp16 False`, otherwise `--fp16 True`.
- `assembly_ai_api_key`: `string` - Your Assembly AI API key. Get yours from [here](https://www.assemblyai.com/app/).

### TTS (Kitten 与 Qwen3 Gradio)

- `tts_backend`: `string` - `kitten` (default, local KittenTTS) or `qwen3` / `qwen3_gradio` / `qwen3_http` (aliases) to call a local [Gradio](https://www.gradio.app/) qwen3-tts style app over HTTP (`gradio_client`).
- `qwen3_tts_url`: `string` - Base URL of the Gradio server (default `http://127.0.0.1:7862`, no trailing slash required).
- `qwen3_tts_http_timeout_seconds`: `number` | `null` - Optional `httpx` tuning for `gradio_client`. **`null` or omit (recommended)** — use Gradio defaults (safest for SSE). If set to a **positive** number, MoneyPrinterV2 passes `httpx.Timeout(connect=30, read=None, write=<value>, pool=120)` so long GPU silences during `/do_job` do not hit a **read** cap (a plain `Timeout(1200)` on all phases can break streaming and look like “TTS 没反应”). `0` or negative: same as omit. For debugging endpoints, run `python scripts/dump_qwen3_gradio_api.py`.
- `qwen3_tts_api_name`: `string` - Which named API to call:
  - **`/do_job_t`** — `predict(text, instruct, chunk_size)`. Does **not** send reference audio or the voices dropdown; only `qwen3_tts_instruct` and `qwen3_tts_gradio_chunk_size` matter among the qwen3-specific knobs.
  - **`/do_job`** — `predict(voices_dropdown, text, prompt_text, prompt_audio, speed, chunk_size, batch, lang, model_type)`. **Requires** `qwen3_tts_reference_audio` to be a path to an existing file on this machine (passed as `prompt_audio`). If you set `/do_job` but the file is missing, the app raises `RuntimeError` (no silent fallback to `/do_job_t`).
  - **`auto`** — Uses `/do_job` when `qwen3_tts_reference_audio` exists, otherwise `/do_job_t`.
- `qwen3_tts_reference_audio`: `string` - Absolute path to reference WAV/MP3 for `/do_job` / `auto` (when you want `/do_job`). Use a short, clean clip that matches the target timbre. For novel chapter segment TTS, an unset or invalid path makes timbre drift more likely when using `/do_job_t`.
- `qwen3_tts_voices_dropdown`: `string` - Must match a **literal** option from the Gradio **音色** dropdown (run `Client(url).view_api()` or open the app’s API page to list values). Examples often include `老男人`, `使用参考音频`, `Keira`, etc. Only used for `/do_job`. When anchoring timbre from `qwen3_tts_reference_audio`, use **`使用参考音频`** (or the option your app uses for “custom reference”) so the server applies your file.
- `qwen3_tts_prompt_text`: `string` - For **`/do_job`**, this should be the **verbatim transcript of what is spoken in `qwen3_tts_reference_audio`** (same wording as in the clip), not a style tagline. If empty, the code falls back to `qwen3_tts_instruct`, which often **breaks** reference cloning and sounds like gibberish. Use the qwen3-tts UI or Gradio **`/prompt_wav_recognition`** (pass the reference file) to obtain text, paste it here, and edit if the ASR is wrong.
- `qwen3_tts_speed`, `qwen3_tts_batch`, `qwen3_tts_lang`, `qwen3_tts_model_type`: passed to `/do_job` only (`model_type`: `0.6B` or `1.7B`). The client wraps the reference path with `gradio_client.handle_file` for `/do_job` (raw paths often make the upstream app error). `batch` is sent as an **integer** (some qwen3-tts builds reject `8.0`).
- `qwen3_tts_instruct`: style / instruction text (`/do_job_t` and as default `prompt_text` for `/do_job`).
- `qwen3_tts_max_chunk_chars`: max characters per chunk before splitting text for repeated `predict` calls.
- `qwen3_tts_gradio_chunk_size`: third slider argument (chunk size) for both `/do_job_t` and `/do_job`.
- `qwen3_tts_static_args` / `qwen3_tts_predict_arg_order`: only used if you point `qwen3_tts_api_name` at a **custom** Gradio endpoint (not `/do_job` or `/do_job_t`).

**`config.json` fragment for Qwen3 + `/do_job`:** set `tts_backend` to `qwen3`, `qwen3_tts_api_name` to `/do_job` or `auto`, and set `qwen3_tts_reference_audio` to a real file path before running.

Optional integration check (local Gradio must be running): `QWEN3_SMOKE=1 .venv/bin/python -m unittest tests.test_tts_qwen3_resolve.Qwen3LiveSmokeTests` — uses a tiny silent WAV; the server may skip or fail if it rejects short clips; use a real reference file for a meaningful end-to-end check.

- `tts_voice`: `string` - Voice for **KittenTTS only** (when `tts_backend` is `kitten`). Default is `Jasper`. Options: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`.
- `font`: `string` - The font that will be used to generate images. This should be a `.ttf` file in the `fonts/` directory.
- `imagemagick_path`: `string` - The path to the ImageMagick binary. This is used by MoviePy to manipulate images. Install ImageMagick from [here](https://imagemagick.org/script/download.php) and set the path to the `magick.exe` on Windows, or on Linux/MacOS the path to `convert` (usually /usr/bin/convert).
- `script_sentence_length`: `number` - The number of sentences in the generated video script (default: `4`).
- `post_bridge`: `object`:
    - `enabled`: `boolean` - Enables Post Bridge cross-posting after successful YouTube uploads.
    - `api_key`: `string` - Your Post Bridge API key. If empty, MPV2 falls back to `POST_BRIDGE_API_KEY`.
    - `platforms`: `string[]` - Platforms to target. Supported values in v1 are `tiktok` and `instagram`.
    - `account_ids`: `number[]` - Optional fixed Post Bridge account IDs to avoid account-selection prompts.
    - `auto_crosspost`: `boolean` - If `true`, cross-post automatically after a successful YouTube upload. If `false`, interactive runs ask and cron runs skip.

## Example

```json
{
  "verbose": true,
  "firefox_profile": "",
  "headless": false,
  "script_api_base_url": "https://api.openai.com/v1",
  "script_api_key": "",
  "script_api_model": "gpt-4.1-mini",
  "twitter_language": "English",
  "nanobanana2_api_base_url": "https://generativelanguage.googleapis.com/v1beta",
  "nanobanana2_api_key": "",
  "nanobanana2_model": "gemini-3.1-flash-image",
  "nanobanana2_aspect_ratio": "9:16",
  "threads": 2,
  "zip_url": "",
  "is_for_kids": false,
  "google_maps_scraper": "https://github.com/gosom/google-maps-scraper/archive/refs/tags/v0.9.7.zip",
  "email": {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "",
    "password": ""
  },
  "google_maps_scraper_niche": "",
  "scraper_timeout": 300,
  "outreach_message_subject": "I have a question...",
  "outreach_message_body_file": "outreach_message.html",
  "stt_provider": "local_whisper",
  "whisper_cli_path": "",
  "whisper_cli_timeout_seconds": 7200,
  "whisper_model": "base",
  "whisper_model_path": "",
  "whisper_device": "auto",
  "whisper_compute_type": "int8",
  "assembly_ai_api_key": "",
  "tts_voice": "Jasper",
  "font": "bold_font.ttf",
  "imagemagick_path": "Path to magick.exe or on linux/macOS just /usr/bin/convert",
  "script_sentence_length": 4,
  "post_bridge": {
    "enabled": false,
    "api_key": "",
    "platforms": ["tiktok", "instagram"],
    "account_ids": [],
    "auto_crosspost": false
  }
}
```

## Environment Variable Fallbacks

- `SCRIPT_API_KEY`: used when `script_api_key` is empty.
- `GEMINI_API_KEY`: used when `nanobanana2_api_key` is empty.
- `POST_BRIDGE_API_KEY`: used when `post_bridge.api_key` is empty.

Example:

```bash
export GEMINI_API_KEY="your_api_key_here"
export POST_BRIDGE_API_KEY="your_post_bridge_api_key_here"
```

See [PostBridge.md](./PostBridge.md) for the full Post Bridge setup and behavior details.
