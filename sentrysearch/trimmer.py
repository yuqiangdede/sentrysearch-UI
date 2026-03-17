"""ffmpeg clip extraction."""

import os
import re
import subprocess

from .chunker import _get_ffmpeg_executable, _get_video_duration


def trim_clip(
    source_file: str,
    start_time: float,
    end_time: float,
    output_path: str,
    padding: float = 2.0,
) -> str:
    """Extract a segment from the original source video using ffmpeg.

    Adds *padding* seconds before and after the match window, clamped to
    file boundaries.  Tries ``-c copy`` first for speed; falls back to
    re-encoding if the copy fails (e.g. when the seek lands mid-GOP).

    Args:
        source_file: Path to the original mp4 file.
        start_time: Match start time in seconds.
        end_time: Match end time in seconds.
        output_path: Where to write the trimmed clip.
        padding: Extra seconds to include before/after the match window.

    Returns:
        The *output_path* on success.
    """
    if end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})."
        )

    duration = _get_video_duration(source_file)
    padded_start = max(0.0, start_time - padding)
    padded_end = min(duration, end_time + padding)
    length = padded_end - padded_start

    ffmpeg_exe = _get_ffmpeg_executable()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Attempt 1: stream-copy (fast, no quality loss)
    copy_result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-ss", str(padded_start),
            "-i", source_file,
            "-t", str(length),
            "-c", "copy",
            output_path,
        ],
        capture_output=True,
        text=True,
    )

    if copy_result.returncode == 0 and os.path.isfile(output_path):
        return output_path

    # Attempt 2: re-encode (handles mid-GOP seeks, format quirks)
    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-ss", str(padded_start),
            "-i", source_file,
            "-t", str(length),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path,
        ],
        capture_output=True,
        check=True,
    )
    return output_path


def _fmt_time(seconds: float) -> str:
    """Format seconds as e.g. '02m15s'."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}m{s:02d}s"


def _safe_filename(source_file: str, start: float, end: float) -> str:
    """Build a filesystem-safe descriptive filename."""
    base = os.path.splitext(os.path.basename(source_file))[0]
    base = re.sub(r"[^\w\-]", "_", base)
    return f"match_{base}_{_fmt_time(start)}-{_fmt_time(end)}.mp4"


def trim_top_result(results: list[dict], output_dir: str) -> str:
    """Trim the highest-ranked search result and save it to *output_dir*.

    Args:
        results: List of result dicts from :func:`search_footage`
                 (must contain source_file, start_time, end_time).
        output_dir: Directory to write the clip into.

    Returns:
        Path to the saved clip.
    """
    if not results:
        raise ValueError("No results to trim.")

    top = results[0]
    filename = _safe_filename(top["source_file"], top["start_time"], top["end_time"])
    output_path = os.path.join(output_dir, filename)

    return trim_clip(
        source_file=top["source_file"],
        start_time=top["start_time"],
        end_time=top["end_time"],
        output_path=output_path,
    )
