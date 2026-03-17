"""Gemini embedding client using the google-genai SDK."""

import os
import sys
import time
from collections import deque

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-embedding-2-preview"
DIMENSIONS = 768
DEFAULT_RPM = 55

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple sliding-window rate limiter based on request timestamps."""

    def __init__(self, max_per_minute: int = DEFAULT_RPM):
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        # Drop timestamps older than 60 s
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


_limiter = _RateLimiter()

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def _retry(fn, *, max_retries: int = 5, initial_delay: float = 2.0, max_delay: float = 60.0):
    """Call *fn* with exponential back-off on transient errors."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

            # Provisioning errors need longer waits and more retries
            provisioning = "service agents are being provisioned" in msg
            retryable = status in (429, 503) or provisioning
            if not retryable:
                retryable = (
                    "resource exhausted" in msg
                    or "503" in msg
                    or "429" in msg
                )
            if not retryable or attempt == max_retries:
                raise

            wait = min(delay, max_delay)
            if provisioning:
                wait = max(wait, 30.0)  # provisioning needs longer waits
            print(
                f"  Retryable error (attempt {attempt + 1}/{max_retries}), "
                f"waiting {wait:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay *= 2

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_video_chunk(chunk_path: str) -> list[float]:
    """Upload a video chunk, embed it, clean up the remote file, return the vector."""
    client = _get_client()

    # Upload
    _limiter.wait()
    video_file = _retry(lambda: client.files.upload(file=chunk_path))

    # Poll until processing is done
    while video_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(
            f"File processing failed with state: {video_file.state.name}"
        )

    # Embed (provisioning can take minutes, allow more retries)
    _limiter.wait()
    response = _retry(
        lambda: client.models.embed_content(
            model=MODEL,
            contents=types.Content(
                parts=[
                    types.Part.from_uri(
                        file_uri=video_file.uri, mime_type="video/mp4"
                    )
                ]
            ),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=DIMENSIONS,
            ),
        ),
        max_retries=8,
    )

    embedding = response.embeddings[0].values

    # Clean up remote file
    try:
        client.files.delete(name=video_file.name)
    except Exception:
        pass  # best-effort cleanup

    return embedding


def embed_query(query_text: str) -> list[float]:
    """Embed a text query for retrieval."""
    client = _get_client()
    _limiter.wait()
    response = _retry(
        lambda: client.models.embed_content(
            model=MODEL,
            contents=query_text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=DIMENSIONS,
            ),
        )
    )
    return response.embeddings[0].values


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed a list of video chunks, printing progress to stderr.

    Args:
        chunks: List of chunk dicts from chunker.chunk_video.

    Returns:
        Same dicts with an added 'embedding' key.
    """
    total = len(chunks)
    results = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Embedding chunk {i}/{total}...", file=sys.stderr)
        embedding = embed_video_chunk(chunk["chunk_path"])
        results.append({**chunk, "embedding": embedding})
    return results
