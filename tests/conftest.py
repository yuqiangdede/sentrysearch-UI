"""Shared test fixtures."""

import math
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Embedder isolation: reset module-level singletons between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_embedder_globals():
    """Reset embedder module-level state so tests don't leak."""
    import sentrysearch.embedder as emb

    original_client = emb._client
    original_limiter = emb._limiter
    emb._client = None
    emb._limiter = emb._RateLimiter(max_per_minute=9999)
    yield
    emb._client = original_client
    emb._limiter = original_limiter


@pytest.fixture(autouse=True)
def _reset_ffmpeg_cache():
    """Clear the lru_cache on _get_ffmpeg_executable between tests."""
    from sentrysearch.chunker import _get_ffmpeg_executable

    _get_ffmpeg_executable.cache_clear()
    yield
    _get_ffmpeg_executable.cache_clear()


# ---------------------------------------------------------------------------
# Mock embedder fixture (returns deterministic 768-dim vectors)
# ---------------------------------------------------------------------------

def _fake_embedding(dim: int = 768) -> list[float]:
    """Return a deterministic unit-normalized embedding vector."""
    vec = [math.sin(i * 0.1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


@pytest.fixture
def mock_embed_query(monkeypatch):
    """Patch embed_query to return a deterministic vector without API calls."""
    fake = _fake_embedding()
    monkeypatch.setattr("sentrysearch.search.embed_query", lambda *a, **kw: fake)
    return fake


@pytest.fixture
def mock_embed_video_chunk(monkeypatch):
    """Patch embed_video_chunk to return a deterministic vector."""
    fake = _fake_embedding()
    monkeypatch.setattr(
        "sentrysearch.embedder.embed_video_chunk", lambda *a, **kw: fake
    )
    return fake


# ---------------------------------------------------------------------------
# Temporary ChromaDB store
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """Create a SentryStore backed by a temporary directory."""
    from sentrysearch.store import SentryStore

    return SentryStore(db_path=tmp_path / "test_db")


# ---------------------------------------------------------------------------
# Tiny synthetic test video (generated via ffmpeg from imageio-ffmpeg)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ffmpeg_exe():
    """Return a working ffmpeg executable path (from imageio-ffmpeg)."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture(scope="session")
def tiny_video(ffmpeg_exe, tmp_path_factory):
    """Generate a 3-second synthetic MP4 video (64x64 @ 10fps)."""
    video_dir = tmp_path_factory.mktemp("videos")
    video_path = video_dir / "test_3s.mp4"
    subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-f", "lavfi",
            "-i", "testsrc2=size=64x64:rate=10:duration=3",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    assert video_path.exists() and video_path.stat().st_size > 0
    return str(video_path)


@pytest.fixture(scope="session")
def longer_video(ffmpeg_exe, tmp_path_factory):
    """Generate a 10-second synthetic MP4 for chunk-splitting tests."""
    video_dir = tmp_path_factory.mktemp("videos")
    video_path = video_dir / "test_10s.mp4"
    subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-f", "lavfi",
            "-i", "testsrc2=size=64x64:rate=10:duration=10",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    assert video_path.exists() and video_path.stat().st_size > 0
    return str(video_path)
