"""Video chunking logic."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def _get_ffmpeg_executable() -> str:
    """Return a usable ffmpeg executable path.

    Search order:
    1. System ffmpeg on PATH
    2. imageio-ffmpeg bundled binary (if installed)
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "ffmpeg not found on PATH and imageio-ffmpeg is not available. "
            "Install ffmpeg system-wide or `pip install imageio-ffmpeg`."
        ) from exc


def _parse_duration_from_ffmpeg_output(stderr_text: str) -> float:
    """Extract duration from ffmpeg stderr line: Duration: HH:MM:SS.xx"""
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text)
    if not match:
        # Surface ffmpeg's own error (e.g. "No such file or directory")
        for line in stderr_text.splitlines():
            lower = line.lower()
            if "error" in lower or "no such file" in lower:
                raise RuntimeError(f"ffmpeg error: {line.strip()}")
        raise RuntimeError("Could not determine video duration from ffmpeg output.")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds.

    Prefer ffprobe when available. Fallback to parsing ffmpeg stderr.
    """
    ffprobe_exe = shutil.which("ffprobe")
    if ffprobe_exe:
        result = subprocess.run(
            [
                ffprobe_exe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

    ffmpeg_exe = _get_ffmpeg_executable()
    result = subprocess.run(
        [ffmpeg_exe, "-i", video_path],
        capture_output=True,
        text=True,
        check=False,
    )
    return _parse_duration_from_ffmpeg_output(result.stderr)


def chunk_video(
    video_path: str,
    chunk_duration: int = 30,
    overlap: int = 5,
) -> list[dict]:
    """Split a video into overlapping chunks using ffmpeg.

    Args:
        video_path: Path to the input mp4 file.
        chunk_duration: Duration of each chunk in seconds.
        overlap: Overlap between consecutive chunks in seconds.

    Returns:
        List of dicts with keys: chunk_path, source_file, start_time, end_time.
    """
    video_path = str(Path(video_path).resolve())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    ffmpeg_exe = _get_ffmpeg_executable()
    duration = _get_video_duration(video_path)
    tmp_dir = tempfile.mkdtemp(prefix="sentrysearch_")
    step = chunk_duration - overlap
    chunks = []

    if duration <= chunk_duration:
        chunk_path = os.path.join(tmp_dir, "chunk_000.mp4")
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-ss", "0",
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",
                chunk_path,
            ],
            capture_output=True,
            check=True,
        )
        return [
            {
                "chunk_path": chunk_path,
                "source_file": video_path,
                "start_time": 0.0,
                "end_time": duration,
            }
        ]

    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_duration, duration)
        t = end - start
        chunk_path = os.path.join(tmp_dir, f"chunk_{idx:03d}.mp4")

        # Input seeking (-ss before -i) for fast seek
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(t),
                "-c", "copy",
                chunk_path,
            ],
            capture_output=True,
            check=True,
        )

        chunks.append(
            {
                "chunk_path": chunk_path,
                "source_file": video_path,
                "start_time": start,
                "end_time": end,
            }
        )

        start += step
        idx += 1

        # Avoid a tiny trailing chunk that's entirely within the previous chunk
        if start + overlap >= duration:
            break

    return chunks


def scan_directory(directory_path: str) -> list[str]:
    """Recursively find all .mp4 files in a directory.

    Args:
        directory_path: Root directory to scan.

    Returns:
        Sorted list of absolute file paths.
    """
    mp4_files = []
    for root, _dirs, files in os.walk(directory_path):
        for f in files:
            if f.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(root, f))
    mp4_files.sort()
    return mp4_files
