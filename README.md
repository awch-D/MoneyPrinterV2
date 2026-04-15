# MoneyPrinterV2 Short Video Generator

This project has been simplified to one purpose: generate a short video with the
pipeline:

`script -> images -> speech -> subtitles -> final mp4`

## Requirements

- Python 3.11
- ImageMagick installed locally (`magick` path configured)
- ffmpeg available (required by MoviePy)
- Script generation API (OpenAI-compatible `/chat/completions`)
- Gemini image API key (Nano Banana model endpoint)

## Installation

```bash
cd MoneyPrinterV2
cp config.example.json config.json
python3.11 -m venv .venv312
.venv312/bin/pip install -r requirements.txt
```

## Configuration

Edit `config.json`:

- `script_api_base_url`, `script_api_key`, `script_api_model`
- `nanobanana2_api_key` (or `GEMINI_API_KEY` env var)
- `imagemagick_path` (for macOS, usually `/opt/homebrew/bin/magick`)
- Optional: `script_sentence_length`, `tts_voice`, `stt_provider`

## Usage

Full CLI (short video + **novel chapter** pipeline, orientation, etc.) is documented here:

- **中文使用说明**: [docs/Usage.md](docs/Usage.md)

Quick start (short / default capability):

```bash
scripts/run.sh src/main.py --niche "technology" --language "English"
```

With a fixed topic:

```bash
scripts/run.sh src/main.py --niche "technology" --language "English" --topic "Why local AI changes creator workflows"
```

The command prints a JSON result containing:

- `video_path`
- `audio_path`
- `subtitle_path`
- `image_paths`
- generated `topic` and `script`

Generated media files are stored in `.mp/`.
