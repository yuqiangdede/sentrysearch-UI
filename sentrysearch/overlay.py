"""Tesla-style metadata overlay for trimmed clips.

Burns speed, location, turn-signal, gear, autopilot status, steering angle,
and brake/pedal information onto a video clip using ffmpeg's ASS subtitle
filter (libass). Styled to closely match the native Tesla dashcam UI.

"""

import functools
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timedelta

from .chunker import _get_ffmpeg_executable, _get_video_duration
from .metadata import extract_metadata


@functools.lru_cache(maxsize=1)
def _get_ass_ffmpeg() -> str:
    """Return an ffmpeg path that supports the ``ass`` subtitle filter.

    The system ffmpeg may be compiled without libass (common on macOS
    homebrew minimal installs). When that happens, fall back to the
    imageio-ffmpeg bundled binary which ships with libass enabled.
    """
    candidate = _get_ffmpeg_executable()
    try:
        r = subprocess.run(
            [candidate, "-filters"],
            capture_output=True, text=True, timeout=5,
        )
        if re.search(r"\bass\b.*V->V", r.stdout):
            return candidate
    except Exception:
        pass

    # system ffmpeg lacks libass, try imageio-ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    # last resort, return what we have and let ffmpeg error naturally
    return candidate


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------


def get_metadata_samples(
    source_file: str,
    start_time: float,
    end_time: float,
    padding: float = 2.0,
) -> list[dict] | None:
    """Return per-second metadata samples spanning the clip's time range.

    Returns ``None`` when the file contains no Tesla SEI data.
    Each returned dict has the raw SEI fields plus a ``clip_offset`` key
    (seconds from clip start).
    """
    all_meta = extract_metadata(source_file)
    if not all_meta:
        return None

    duration = _get_video_duration(source_file)
    if duration <= 0:
        return None

    n_frames = len(all_meta)
    padded_start = max(0.0, start_time - padding)
    padded_end = min(duration, end_time + padding)
    clip_duration = padded_end - padded_start

    samples = []
    t = 0.0
    while t <= clip_duration:
        source_t = padded_start + t
        fraction = max(0.0, min(source_t / duration, 1.0))
        idx = min(int(n_frames * fraction), n_frames - 1)
        sample = dict(all_meta[idx])
        sample["clip_offset"] = t
        samples.append(sample)
        t += 1.0

    return samples if samples else None


def reverse_geocode(lat: float, lon: float) -> dict | None:
    """Reverse-geocode *lat*/*lon* into ``{"city": ..., "road": ...}``.

    Uses geopy + Nominatim.  Returns ``None`` when geopy is not installed
    or the lookup fails.  Results are cached by coordinates rounded to
    4 decimal places (~11 m precision).
    """
    try:
        from geopy.geocoders import Nominatim  # noqa: F811
    except ImportError:
        return None

    rounded = (round(lat, 4), round(lon, 4))
    return _geocode_cached(rounded)


@functools.lru_cache(maxsize=64)
def _geocode_cached(coords: tuple[float, float]) -> dict | None:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderServiceError

    time.sleep(1)  # respect Nominatim rate limit
    try:
        geolocator = Nominatim(user_agent="sentrysearch")
        location = geolocator.reverse(coords, language="en", timeout=5)
    except (GeocoderServiceError, Exception):
        return None

    if location is None:
        return None

    addr = location.raw.get("address", {})
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("county", "")
    )
    road = addr.get("road", "")
    return {"city": city, "road": road}


# ---------------------------------------------------------------------------
# overlay rendering
# ---------------------------------------------------------------------------

_AUTOPILOT_LABELS = {
    "NONE": "",
    "SELF_DRIVING": "FSD",
    "AUTOSTEER": "Autosteer",
    "TACC": "TACC",
}

_GEAR_LABELS = {
    "GEAR_PARK": "P",
    "GEAR_DRIVE": "D",
    "GEAR_REVERSE": "R",
    "GEAR_NEUTRAL": "N",
}

# ASS colour constants (&HAABBGGRR format)
_WHITE = "&H00FFFFFF"
_LIGHT_GRAY = "&H00CCCCCC"
_DIM_GRAY = "&H00888888"
_GREEN = "&H0000CC00"
_RED = "&H000000FF"
_SHADOW_COL = "&H80000000"
_AP_BLUE = "&H00FF8800"


def _parse_base_datetime(source_file: str) -> datetime | None:
    """Parse the base datetime from a Tesla filename, ffmpeg metadata, or file mtime."""
    basename = os.path.basename(source_file)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", basename)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
            )
        except ValueError:
            pass

    try:
        ffmpeg_exe = _get_ffmpeg_executable()
        r = subprocess.run(
            [ffmpeg_exe, "-i", source_file],
            capture_output=True, text=True,
        )
        ct = re.search(
            r"creation_time\s*:\s*(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})",
            r.stderr,
        )
        if ct:
            raw = ct.group(1).replace("T", " ")
            return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    try:
        mtime = os.path.getmtime(source_file)
        return datetime.fromtimestamp(mtime)
    except OSError:
        return None


def _format_datetime(dt: datetime) -> str:
    """Format datetime as ``YYYY-MM-DD  HH:MM AM/PM``."""
    return dt.strftime("%Y-%m-%d  %I:%M %p")


def _secs_to_ass_time(s: float) -> str:
    """Convert seconds to ASS timestamp ``H:MM:SS.cc``."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}:{m:02d}:{sec:05.2f}"


def _get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Return (width, height) of a video using ffmpeg."""
    ffmpeg_exe = _get_ffmpeg_executable()
    r = subprocess.run(
        [ffmpeg_exe, "-i", video_path],
        capture_output=True, text=True,
    )
    m = re.search(r"(\d{2,5})x(\d{2,5})", r.stderr)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 1280, 960


def _chevron_right(scale: float) -> str:
    """Right-pointing turn signal chevron (>), ASS drawing commands."""
    s = scale
    w, h = int(10 * s), int(16 * s)
    return f"m {-w} {-h} l {w} 0 {-w} {h}"


def _chevron_left(scale: float) -> str:
    """Left-pointing turn signal chevron (<), ASS drawing commands."""
    s = scale
    w, h = int(10 * s), int(16 * s)
    return f"m {w} {-h} l {-w} 0 {w} {h}"


def _build_ass_content(
    samples: list[dict],
    clip_duration: float,
    location_line: str,
    base_dt: datetime | None,
    start_offset: float,
    video_width: int,
    video_height: int,
) -> str:
    """Generate ASS subtitle content styled to match native Tesla dashcam UI.

    Layout (top-center):
        [left blinker]  SPEED  [right blinker]
                         mph
                  Gear | AP Status
                    DateTime
        Location (top-left)
        Steering | Brake/Accel (bottom-left)
    """
    scale = min(video_width / 1280, video_height / 720)
    cx = video_width // 2

    # vertical positions (from top)
    speed_y = int(38 * scale)
    mph_y = speed_y + int(30 * scale)
    status_y = mph_y + int(20 * scale)
    dt_y = status_y + int(20 * scale)

    # turn signal positions (flanking speed)
    blinker_offset = int(80 * scale)
    blinker_y = speed_y + int(4 * scale)

    # bottom-left info block
    bl_x = int(20 * scale)
    bl_y = video_height - int(24 * scale)

    # font sizes
    speed_fs = int(38 * scale)
    mph_fs = int(13 * scale)
    status_fs = int(13 * scale)
    dt_fs = int(12 * scale)
    loc_fs = int(14 * scale)
    info_fs = int(12 * scale)

    end_all = _secs_to_ass_time(clip_duration + 1)

    # ASS styles: white text with subtle dark shadow for readability
    styles = [
        # Main text style: white, slight shadow
        f"Style: HUD,Arial,{speed_fs},{_WHITE},&H00FFFFFF,"
        f"{_SHADOW_COL},{_SHADOW_COL},0,0,0,0,100,100,0,0,"
        f"1,0,1.5,5,0,0,0,1",
        # Location style: white with dark outline for contrast
        f"Style: Loc,Arial,{loc_fs},{_WHITE},&H00FFFFFF,"
        f"&H60000000,&H80000000,0,0,0,0,100,100,0,0,"
        f"1,1.2,0.8,7,{int(16 * scale)},0,{int(12 * scale)},1",
        # Drawing style for shapes (blinkers, indicators)
        f"Style: Draw,Arial,20,{_WHITE},&H00FFFFFF,"
        f"&H00000000,&H00000000,0,0,0,0,100,100,0,0,"
        f"1,0,0,5,0,0,0,1",
    ]

    events = []

    def _ev(layer, start, end, style, text):
        events.append(
            f"Dialogue: {layer},{start},{end},{style},,0,0,0,,{text}"
        )

    # --- static elements (full duration) --------------------------------------
    if location_line:
        _ev(1, "0:00:00.00", end_all, "Loc", location_line)

    # --- per-second dynamic elements ----------------------------------------
    for i, sample in enumerate(samples):
        t_start = sample["clip_offset"]
        t_end = (
            samples[i + 1]["clip_offset"]
            if i + 1 < len(samples)
            else clip_duration + 1
        )
        ass_s = _secs_to_ass_time(t_start)
        ass_e = _secs_to_ass_time(t_end)

        # -- Speed (large, centered) ----------------------------------------
        speed_mps = sample.get("vehicle_speed_mps", 0) or 0
        speed_mph = int(round(speed_mps * 2.23694))
        _ev(
            2, ass_s, ass_e, "HUD",
            f"{{\\an5\\pos({cx},{speed_y})\\fs{speed_fs}\\b1"
            f"\\c{_WHITE}\\bord0\\shad1.5\\4c{_SHADOW_COL}}}{speed_mph}",
        )

        # -- "mph" label (below speed) --------------------------------------
        _ev(
            2, ass_s, ass_e, "HUD",
            f"{{\\an5\\pos({cx},{mph_y})\\fs{mph_fs}\\b0"
            f"\\c{_LIGHT_GRAY}\\bord0\\shad1\\4c{_SHADOW_COL}}}mph",
        )

        # -- Turn signals (green chevrons flanking speed) -------------------
        left_on = sample.get("blinker_on_left", False)
        right_on = sample.get("blinker_on_right", False)

        left_col = _GREEN if left_on else _DIM_GRAY
        right_col = _GREEN if right_on else _DIM_GRAY
        left_alpha = "" if left_on else "\\1a&H90&"
        right_alpha = "" if right_on else "\\1a&H90&"

        _ev(
            3, ass_s, ass_e, "Draw",
            f"{{\\an6\\pos({cx - blinker_offset + int(8 * scale)},{blinker_y})"
            f"\\c{left_col}{left_alpha}\\bord0\\shad0"
            f"\\p1}}{_chevron_left(scale)}",
        )
        _ev(
            3, ass_s, ass_e, "Draw",
            f"{{\\an4\\pos({cx + blinker_offset},{blinker_y})"
            f"\\c{right_col}{right_alpha}\\bord0\\shad0"
            f"\\p1}}{_chevron_right(scale)}",
        )

        # -- Gear + Autopilot status row ------------------------------------
        gear_state = sample.get("gear_state", "GEAR_PARK") or "GEAR_PARK"
        gear_label = _GEAR_LABELS.get(gear_state, "D")
        ap_state = sample.get("autopilot_state", "NONE") or "NONE"
        ap_label = _AUTOPILOT_LABELS.get(ap_state, "")

        # Gear character
        gear_col = _WHITE
        _ev(
            2, ass_s, ass_e, "HUD",
            f"{{\\an5\\pos({cx},{status_y})\\fs{status_fs}\\b1"
            f"\\c{gear_col}\\bord0\\shad1\\4c{_SHADOW_COL}}}"
            f"{gear_label}"
            + (
                f"  {{\\c{_AP_BLUE}\\b0}}{ap_label}"
                if ap_label
                else ""
            ),
        )

        # -- DateTime -------------------------------------------------------
        if base_dt:
            dt = base_dt + timedelta(seconds=start_offset + t_start)
            _ev(
                2, ass_s, ass_e, "HUD",
                f"{{\\an5\\pos({cx},{dt_y})\\fs{dt_fs}\\b0"
                f"\\c{_DIM_GRAY}\\bord0\\shad1\\4c{_SHADOW_COL}}}"
                f"{_format_datetime(dt)}",
            )

        # -- Bottom-left: steering angle + brake/accel ----------------------
        steer_angle = sample.get("steering_wheel_angle", 0) or 0
        brake_on = sample.get("brake_applied", False)
        accel_pct = sample.get("accelerator_pedal_position", 0) or 0

        info_parts = []
        # Steering
        direction = "L" if steer_angle < 0 else "R"
        abs_angle = abs(steer_angle)
        if abs_angle >= 0.5:
            info_parts.append(f"{abs_angle:.0f}\\u00B0 {direction}")

        # Brake
        if brake_on:
            info_parts.append(
                f"{{\\c{_RED}\\b1}}BRAKE{{\\c{_WHITE}\\b0}}"
            )

        # Accelerator (only show when pressing > 5%)
        if accel_pct > 0.05 and not brake_on:
            pct_display = int(round(accel_pct * 100))
            info_parts.append(
                f"{{\\c{_GREEN}\\b0}}{pct_display}% throttle"
                f"{{\\c{_WHITE}}}"
            )

        if info_parts:
            info_text = "   ".join(info_parts)
            _ev(
                2, ass_s, ass_e, "HUD",
                f"{{\\an1\\pos({bl_x},{bl_y})\\fs{info_fs}\\b0"
                f"\\c{_WHITE}\\bord0\\shad1.2\\4c{_SHADOW_COL}}}"
                f"{info_text}",
            )

    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(styles)
        + "\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
        + "\n".join(events)
        + "\n"
    )


def apply_overlay(
    input_path: str,
    output_path: str,
    samples: list[dict],
    location: dict | None = None,
    source_file: str | None = None,
    start_time: float = 0.0,
    padding: float = 2.0,
) -> str:
    """Burn a Tesla-style HUD overlay onto a video clip.

    Uses per-second ASS subtitle events so all telemetry fields update
    throughout the clip. Returns *output_path* on success, *input_path*
    on failure.
    """
    ffmpeg_exe = _get_ass_ffmpeg()
    width, height = _get_video_dimensions(input_path)
    clip_duration = _get_video_duration(input_path)

    city, road = "", ""
    if location:
        city = location.get("city", "")
        road = location.get("road", "")
    location_line = f"{city} | {road}" if city and road else city or road

    base_dt = _parse_base_datetime(source_file) if source_file else None
    padded_start = max(0.0, start_time - padding)

    ass_content = _build_ass_content(
        samples=samples,
        clip_duration=clip_duration,
        location_line=location_line,
        base_dt=base_dt,
        start_offset=padded_start,
        video_width=width,
        video_height=height,
    )

    ass_fd, ass_path = tempfile.mkstemp(suffix=".ass", prefix="sentrysearch_")
    try:
        with os.fdopen(ass_fd, "w") as f:
            f.write(ass_content)

        escaped = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        vf = f"ass={escaped}"

        for codec_args in (
            ["-c:v", "libx264", "-crf", "18"],
            ["-c:v", "mpeg4", "-q:v", "5"],
        ):
            result = subprocess.run(
                [
                    ffmpeg_exe, "-y",
                    "-i", input_path,
                    "-vf", vf,
                    *codec_args,
                    "-c:a", "copy",
                    output_path,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and os.path.isfile(output_path):
                return output_path
    finally:
        try:
            os.unlink(ass_path)
        except OSError:
            pass

    return input_path
