"""Tesla-style metadata overlay for trimmed clips.

Burns speed, location, turn-signal, and driving-mode information onto
a video clip using ffmpeg's ASS subtitle filter (libass).

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
    """Reverse-geocode *lat*/*lon* into ``{"city": …, "road": …}``.

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
    "SELF_DRIVING": "Self Driving",
    "AUTOSTEER": "Autosteer",
    "TACC": "TACC",
}

_GEAR_LABELS = {
    "GEAR_PARK": "P",
    "GEAR_DRIVE": "D",
    "GEAR_REVERSE": "R",
    "GEAR_NEUTRAL": "N",
}

# ASS bezier circle (radius 14 at scale 1.0) centered at origin.
# Control-point factor 0.552 gives a good circular approximation.
_CIRCLE_R = 14
_CIRCLE_CP = 8  # int(14 * 0.552)
_CIRCLE_DRAW = (
    f"m 0 -{_CIRCLE_R} "
    f"b {_CIRCLE_CP} -{_CIRCLE_R} {_CIRCLE_R} -{_CIRCLE_CP} {_CIRCLE_R} 0 "
    f"b {_CIRCLE_R} {_CIRCLE_CP} {_CIRCLE_CP} {_CIRCLE_R} 0 {_CIRCLE_R} "
    f"b -{_CIRCLE_CP} {_CIRCLE_R} -{_CIRCLE_R} {_CIRCLE_CP} -{_CIRCLE_R} 0 "
    f"b -{_CIRCLE_R} -{_CIRCLE_CP} -{_CIRCLE_CP} -{_CIRCLE_R} 0 -{_CIRCLE_R}"
)

# Turn signal arrow (chevron pointing right, ~20px wide, ~28px tall)
_ARROW_R_DRAW = "m 0 -12 l 10 0 0 12 -3 12 7 0 -3 -12"
# Left arrow: same shape mirrored
_ARROW_L_DRAW = "m 0 -12 l -10 0 0 12 3 12 -7 0 3 -12"


def _parse_base_datetime(source_file: str) -> datetime | None:
    """Parse the base datetime from a Tesla filename, ffmpeg metadata, or file mtime."""
    # 1. Try Tesla filename pattern (e.g. 2024-01-15_14-30-00-front.mp4)
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

    # 2. Try ffmpeg creation_time metadata
    try:
        ffmpeg_exe = _get_ffmpeg_executable()
        r = subprocess.run(
            [ffmpeg_exe, "-i", source_file],
            capture_output=True, text=True,
        )
        ct = re.search(r"creation_time\s*:\s*(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", r.stderr)
        if ct:
            raw = ct.group(1).replace("T", " ")
            return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    # 3. Fall back to file modification time
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


def _scaled_circle(scale: float) -> str:
    """Return ASS bezier circle drawing commands scaled appropriately."""
    r = int(14 * scale)
    cp = int(8 * scale)
    return (
        f"m 0 -{r} "
        f"b {cp} -{r} {r} -{cp} {r} 0 "
        f"b {r} {cp} {cp} {r} 0 {r} "
        f"b -{cp} {r} -{r} {cp} -{r} 0 "
        f"b -{r} -{cp} -{cp} -{r} 0 -{r}"
    )


def _scaled_arrow_r(scale: float) -> str:
    """Right-pointing chevron arrow, matching ExportDash blinker size."""
    s = scale
    # Wider, taller arrow: 18px wide, 30px tall
    return (
        f"m 0 {int(-15*s)} "
        f"l {int(18*s)} 0 "
        f"0 {int(15*s)} "
        f"{int(-5*s)} {int(15*s)} "
        f"{int(13*s)} 0 "
        f"{int(-5*s)} {int(-15*s)}"
    )


def _scaled_arrow_l(scale: float) -> str:
    """Left-pointing chevron arrow, matching ExportDash blinker size."""
    s = scale
    return (
        f"m 0 {int(-15*s)} "
        f"l {int(-18*s)} 0 "
        f"0 {int(15*s)} "
        f"{int(5*s)} {int(15*s)} "
        f"{int(-13*s)} 0 "
        f"{int(5*s)} {int(-15*s)}"
    )


def _build_ass_content(
    samples: list[dict],
    clip_duration: float,
    gear_label: str,
    mode_label: str,
    brake_applied: bool,
    autopilot_active: bool,
    location_line: str,
    base_dt: datetime | None,
    start_offset: float,
    video_width: int,
    video_height: int,
) -> str:
    """Generate ASS subtitle content matching the ExportDash 2-row layout.

    """
    scale = min(video_width / 1280, video_height / 720)
    cx = video_width // 2

    # -- card geometry -------------------------------------------------------
    card_w = int(140 * scale)
    card_h = int(65 * scale)
    card_y = int(36 * scale)
    card_left = cx - card_w // 2

    # Speed + MPH centered in the card
    speed_cy = card_y + int(24 * scale)
    mph_cy = card_y + int(48 * scale)

    # Font sizes
    speed_fs = int(36 * scale)
    mph_fs = int(14 * scale)
    datetime_fs = int(14 * scale)
    location_fs = int(16 * scale)

    # Colours (ASS &HAABBGGRR)
    col_dark = "&H00333333"
    col_white = "&H00FFFFFF"
    col_datetime = "&H19FFFFFF"

    end_all = _secs_to_ass_time(clip_duration + 1)

    # -- styles ---------------------------------------------------------------
    styles = [
        "Style: D,Arial,20,&H00FFFFFF,&H00FFFFFF,"
        "&H00000000,&H00000000,0,0,0,0,100,100,0,0,"
        "1,0,0,5,0,0,0,1",
        f"Style: Loc,Arial,{location_fs},{col_white},&H00FFFFFF,"
        f"&H50000000,&H80000000,0,0,0,0,100,100,0,0,"
        f"1,1.5,1,7,{int(16 * scale)},0,{int(12 * scale)},1",
    ]

    events = []

    def _ev(layer, start, end, style, text):
        events.append(f"Dialogue: {layer},{start},{end},{style},,0,0,0,,{text}")

    # --- static elements (full duration) ------------------------------------

    # Location (top-left)
    if location_line:
        _ev(1, "0:00:00.00", end_all, "Loc", location_line)

    # "MPH" label — centered in card, below speed
    _ev(1, "0:00:00.00", end_all, "D",
        f"{{\\an5\\pos({cx},{mph_cy})\\fs{mph_fs}\\b1\\c{col_dark}\\bord0\\shad0}}MPH")

    # --- per-second elements (speed, datetime) ------------------------------
    for i, sample in enumerate(samples):
        t_start = sample["clip_offset"]
        t_end = samples[i + 1]["clip_offset"] if i + 1 < len(samples) else clip_duration + 1
        ass_s = _secs_to_ass_time(t_start)
        ass_e = _secs_to_ass_time(t_end)

        # Speed number — centered in card
        speed_mps = sample.get("vehicle_speed_mps", 0) or 0
        speed_mph = int(round(speed_mps * 2.23694))
        _ev(1, ass_s, ass_e, "D",
            f"{{\\an5\\pos({cx},{speed_cy})\\fs{speed_fs}\\b1\\c{col_dark}\\bord0\\shad0}}{speed_mph}")

        # DateTime pill — below card
        if base_dt:
            dt = base_dt + timedelta(seconds=start_offset + t_start)
            dt_y = card_y + card_h + int(8 * scale)
            _ev(1, ass_s, ass_e, "D",
                f"{{\\an8\\pos({cx},{dt_y})\\fs{datetime_fs}\\b0\\c{col_datetime}"
                f"\\bord6\\shad0\\3c&H66000000&\\4c&H66000000&}}{_format_datetime(dt)}")

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
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
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

    Uses per-second ASS subtitle events so speed and time update throughout
    the clip.  Returns *output_path* on success, *input_path* on failure.
    """
    ffmpeg_exe = _get_ffmpeg_executable()
    width, height = _get_video_dimensions(input_path)
    scale = min(width / 1280, height / 720)
    clip_duration = _get_video_duration(input_path)

    mid = samples[len(samples) // 2]

    gear_state = mid.get("gear_state", "GEAR_PARK") or "GEAR_PARK"
    gear_label = _GEAR_LABELS.get(gear_state, "D")
    ap_state = mid.get("autopilot_state", "NONE") or "NONE"
    mode_label = _AUTOPILOT_LABELS.get(ap_state, "")
    brake_applied = mid.get("brake_applied", False)
    autopilot_active = ap_state != "NONE"

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
        gear_label=gear_label,
        mode_label=mode_label,
        brake_applied=brake_applied,
        autopilot_active=autopilot_active,
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

        # drawbox: light gray card background behind speed
        cx = width // 2
        card_w = int(140 * scale)
        card_h = int(65 * scale)
        card_y = int(36 * scale)
        card_x = cx - card_w // 2
        vf = (
            f"drawbox=x={card_x}:y={card_y}:w={card_w}:h={card_h}"
            f":color=0xE1E1E1@0.85:t=fill,"
            f"ass='{ass_path}'"
        )

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
