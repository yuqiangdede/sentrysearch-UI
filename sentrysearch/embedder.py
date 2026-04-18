"""Embedder factory - selects and caches the active backend.

Provides backward-compatible top-level functions (embed_video_chunk,
embed_query) that delegate to whichever backend is currently active.
Re-exports error classes from gemini_embedder for existing import sites.
"""

from .base_embedder import BaseEmbedder
from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError  # noqa: F401

_embedder_cache: dict[tuple, BaseEmbedder] = {}


def _embedder_key(backend: str, kwargs: dict) -> tuple:
    """Build a stable cache key for the selected backend and config."""
    if backend == "gemini":
        return ("gemini",)
    if backend == "local":
        model = kwargs.get("model", "qwen2b")
        dimensions = kwargs.get("dimensions", 768)
        quantize = kwargs.get("quantize", None)
        return ("local", model, dimensions, quantize)
    return (backend, tuple(sorted(kwargs.items())))


def get_embedder(backend: str = "gemini", **kwargs) -> BaseEmbedder:
    """Factory to get or create the active embedder."""
    key = _embedder_key(backend, kwargs)
    cached = _embedder_cache.get(key)
    if cached is not None:
        return cached

    # Keep only one active embedder in memory so switching configs does not
    # leave multiple large models resident at once.
    _embedder_cache.clear()

    if backend == "gemini":
        from .gemini_embedder import GeminiEmbedder

        embedder = GeminiEmbedder()
    elif backend == "local":
        from .local_embedder import LocalEmbedder

        model = kwargs.get("model", "qwen2b")
        dims = kwargs.get("dimensions", 768)
        quantize = kwargs.get("quantize", None)
        embedder = LocalEmbedder(model_name=model, dimensions=dims, quantize=quantize)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    _embedder_cache[key] = embedder
    return embedder


def reset_embedder():
    """Reset the cached embedder (for switching backends)."""
    _embedder_cache.clear()


def _get_active_embedder(backend: str = "gemini", **kwargs) -> BaseEmbedder:
    """Return the current embedder if one is cached, otherwise create one."""
    if len(_embedder_cache) == 1:
        return next(iter(_embedder_cache.values()))
    return get_embedder(backend, **kwargs)


# Convenience functions - backward compatible with existing callers
def embed_video_chunk(chunk_path: str, verbose: bool = False) -> list[float]:
    return _get_active_embedder().embed_video_chunk(chunk_path, verbose=verbose)


def embed_query(query_text: str, verbose: bool = False) -> list[float]:
    return _get_active_embedder().embed_query(query_text, verbose=verbose)
