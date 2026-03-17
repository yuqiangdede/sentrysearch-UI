# sentrysearch

Search dashcam footage using natural language queries.

## Setup

```bash
cp .env.example .env
# Edit .env with your Gemini API key
pip install -e .
```

## Usage

```bash
# Index video files
sentrysearch index /path/to/dashcam/footage

# Search indexed footage
sentrysearch search "red truck running a stop sign"

# Trim a clip from a result
sentrysearch trim source.mp4 --start 30 --end 45 -o clip.mp4
```

## Requirements

- Python 3.10+
- `ffmpeg` on PATH, or use bundled ffmpeg via `imageio-ffmpeg` (installed by default)
