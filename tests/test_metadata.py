"""Tests for sentrysearch.metadata."""

import pytest

from sentrysearch.metadata import _strip_emulation_prevention_bytes, extract_metadata


class TestStripEmulationPreventionBytes:
    def test_no_prevention_bytes(self):
        data = b"\x01\x02\x03\x04"
        assert _strip_emulation_prevention_bytes(data) == data

    def test_removes_prevention_byte(self):
        data = b"\x00\x00\x03\x01"
        assert _strip_emulation_prevention_bytes(data) == b"\x00\x00\x01"

    def test_preserves_non_prevention_sequences(self):
        data = b"\x00\x00\x04"
        assert _strip_emulation_prevention_bytes(data) == b"\x00\x00\x04"


class TestExtractMetadata:
    def test_non_mp4_returns_empty(self, tmp_path):
        fake = tmp_path / "not_mp4.txt"
        fake.write_text("hello")
        assert extract_metadata(str(fake)) == []

    def test_nonexistent_file_returns_empty(self):
        assert extract_metadata("/nonexistent/file.mp4") == []

    def test_synthetic_video_no_sei(self, tiny_video):
        assert extract_metadata(tiny_video) == []
