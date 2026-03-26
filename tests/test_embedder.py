"""Tests for sentrysearch.embedder."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.embedder import (
    GeminiAPIKeyError,
    GeminiQuotaError,
    _RateLimiter,
    _get_client,
    _retry,
    embed_query,
    embed_video_chunk,
)


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        limiter = _RateLimiter(max_per_minute=5)
        for _ in range(5):
            limiter.wait()

    def test_tracks_request_count(self):
        limiter = _RateLimiter(max_per_minute=3)
        for _ in range(3):
            limiter.wait()
        assert len(limiter._timestamps) == 3

    @patch("sentrysearch.embedder.time.sleep")
    @patch("sentrysearch.embedder.time.monotonic")
    def test_sleeps_when_limit_reached(self, mock_monotonic, mock_sleep):
        limiter = _RateLimiter(max_per_minute=2)
        # First two at t=0 and t=1
        mock_monotonic.return_value = 0.0
        limiter.wait()
        mock_monotonic.return_value = 1.0
        limiter.wait()
        # Third at t=2: window still has 2 requests
        mock_monotonic.return_value = 2.0
        limiter.wait()
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] > 0

    def test_window_slides(self):
        limiter = _RateLimiter(max_per_minute=1)
        limiter._timestamps.append(time.monotonic() - 61)  # expired
        limiter.wait()  # should not block
        assert len(limiter._timestamps) == 1


# ---------------------------------------------------------------------------
# _get_client
# ---------------------------------------------------------------------------

class TestGetClient:
    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(GeminiAPIKeyError, match="GEMINI_API_KEY"):
                _get_client()

    @patch("sentrysearch.embedder.genai.Client")
    def test_creates_client_with_key(self, mock_client_cls):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
            import sentrysearch.embedder as emb
            emb._client = None  # force re-creation
            client = _get_client()
            mock_client_cls.assert_called_once_with(api_key="test-key-123")


# ---------------------------------------------------------------------------
# _retry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_returns_on_first_success(self):
        fn = MagicMock(return_value="ok")
        assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"
        fn.assert_called_once()

    @patch("sentrysearch.embedder.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        exc = Exception("Resource exhausted")
        exc.status_code = 429
        fn = MagicMock(side_effect=[exc, exc, "ok"])
        assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"
        assert fn.call_count == 3

    @patch("sentrysearch.embedder.time.sleep")
    def test_retries_on_503(self, mock_sleep):
        exc = Exception("Service unavailable")
        exc.status_code = 503
        fn = MagicMock(side_effect=[exc, "ok"])
        assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"

    @patch("sentrysearch.embedder.time.sleep")
    def test_raises_quota_error_after_max_retries(self, mock_sleep):
        exc = Exception("resource exhausted")
        exc.status_code = 429
        fn = MagicMock(side_effect=exc)
        with pytest.raises(GeminiQuotaError):
            _retry(fn, max_retries=2, initial_delay=0.01)

    def test_raises_non_retryable_immediately(self):
        fn = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            _retry(fn, max_retries=3, initial_delay=0.01)

    @patch("sentrysearch.embedder.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        exc = Exception("503 error")
        exc.status_code = 503
        fn = MagicMock(side_effect=[exc, exc, "ok"])
        _retry(fn, max_retries=3, initial_delay=1.0)
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays[0] == 1.0
        assert delays[1] == 2.0


# ---------------------------------------------------------------------------
# embed_query / embed_video_chunk with mocked client
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    @patch("sentrysearch.embedder._get_client")
    def test_returns_vector(self, mock_get_client):
        fake_values = [0.1] * 768
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=fake_values)]
        mock_client.models.embed_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = embed_query("a red car")
        assert result == fake_values
        assert len(result) == 768
        mock_client.models.embed_content.assert_called_once()


class TestEmbedVideoChunk:
    @patch("sentrysearch.embedder._get_client")
    def test_returns_vector(self, mock_get_client, tiny_video):
        fake_values = [0.2] * 768
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=fake_values)]
        mock_client.models.embed_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = embed_video_chunk(tiny_video)
        assert result == fake_values
        assert len(result) == 768
