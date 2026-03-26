"""Tests for sentrysearch.trimmer."""

import os

import pytest

from sentrysearch.trimmer import _fmt_time, _safe_filename, trim_clip, trim_top_result


class TestFmtTime:
    def test_zero(self):
        assert _fmt_time(0) == "00m00s"

    def test_seconds(self):
        assert _fmt_time(45) == "00m45s"

    def test_minutes_and_seconds(self):
        assert _fmt_time(125) == "02m05s"

    def test_float_truncates(self):
        assert _fmt_time(59.9) == "00m59s"


class TestSafeFilename:
    def test_basic(self):
        result = _safe_filename("/path/to/video.mp4", 10.0, 40.0)
        assert result.startswith("match_")
        assert result.endswith(".mp4")
        assert "00m10s" in result
        assert "00m40s" in result

    def test_special_chars_cleaned(self):
        result = _safe_filename("/path/to/my video (1).mp4", 0.0, 30.0)
        assert " " not in result
        assert "(" not in result


class TestTrimClip:
    def test_trims_clip(self, tiny_video, tmp_path):
        output = str(tmp_path / "trimmed.mp4")
        result = trim_clip(tiny_video, 0.5, 2.5, output, padding=0.5)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_end_before_start_raises(self, tiny_video, tmp_path):
        output = str(tmp_path / "bad.mp4")
        with pytest.raises(ValueError, match="end_time.*must be greater"):
            trim_clip(tiny_video, 10.0, 5.0, output)

    def test_creates_output_directory(self, tiny_video, tmp_path):
        deep_output = str(tmp_path / "a" / "b" / "c" / "clip.mp4")
        result = trim_clip(tiny_video, 0.5, 2.0, deep_output)
        assert os.path.isfile(result)


class TestTrimTopResult:
    def test_trims_first_result(self, tiny_video, tmp_path):
        results = [
            {
                "source_file": tiny_video,
                "start_time": 0.5,
                "end_time": 2.5,
                "similarity_score": 0.95,
            },
        ]
        clip = trim_top_result(results, str(tmp_path))
        assert os.path.isfile(clip)
        assert clip.startswith(str(tmp_path))

    def test_empty_results_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No results"):
            trim_top_result([], str(tmp_path))
