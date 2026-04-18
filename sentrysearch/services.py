"""Shared service layer for CLI and web workflows."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable

from .chunker import (
    SUPPORTED_VIDEO_EXTENSIONS,
    chunk_video,
    is_still_frame_chunk,
    preprocess_chunk,
    scan_directory,
)
from .embedder import get_embedder
from .local_embedder import detect_default_model, normalize_model_key
from .overlay import apply_overlay, get_metadata_samples, reverse_geocode
from .search import search_footage
from .reranker import get_reranker
from .paths import CLIPS_DIR, resolve_project_path, ensure_dir
from .store import SentryStore, detect_index
from .trimmer import trim_clip

ProgressCallback = Callable[[dict], None]


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _safe_filename(source_file: str, start: float, end: float) -> str:
    base = os.path.splitext(os.path.basename(source_file))[0]
    base = re.sub(r"[^\w\-]", "_", base)
    return f"match_{base}_{_fmt_time(start).replace(':', 'm')}-{_fmt_time(end).replace(':', 'm')}.mp4"


def resolve_index_backend_model(
    backend: str | None = None,
    model: str | None = None,
) -> tuple[str, str | None]:
    """Resolve backend/model for indexing."""
    if model is not None and backend is None:
        backend = "local"
    if backend is None:
        backend = "gemini"
    if backend == "local":
        if model is None:
            model = detect_default_model()
        model = normalize_model_key(model)
    return backend, model


def resolve_search_backend_model(
    backend: str | None = None,
    model: str | None = None,
) -> tuple[str, str | None]:
    """Resolve backend/model for search based on index metadata."""
    if model is not None and backend is None:
        backend = "local"
    if model is not None:
        model = normalize_model_key(model)

    if backend is None:
        detected_backend, detected_model = detect_index()
        backend = detected_backend or "gemini"
        if model is None:
            model = detected_model
    elif backend == "local" and model is None:
        _, detected_model = detect_index()
        model = detected_model
    return backend, model


def get_stats(backend: str | None = None, model: str | None = None) -> dict:
    """Return index statistics with backend/model info."""
    try:
        if model is not None and backend is None:
            backend = "local"
        if backend is None:
            backend, detected_model = detect_index()
            backend = backend or "gemini"
            if model is None:
                model = detected_model

        store = SentryStore(backend=backend, model=model)
        stats = store.get_stats()
        return {
            **stats,
            "backend": store.get_backend(),
            "model": store.get_model(),
        }
    except Exception:
        return {
            "total_chunks": 0,
            "unique_source_files": 0,
            "source_files": [],
            "backend": backend or "gemini",
            "model": model,
        }


def get_video_index_status(
    source_files: list[str],
    *,
    backend: str | None = None,
    model: str | None = None,
) -> dict:
    """Return index status mapping for specific source files."""
    normalized = [os.path.abspath(p) for p in source_files]
    backend, model = resolve_index_backend_model(backend, model)
    try:
        store = SentryStore(backend=backend, model=model)
        status = {path: store.is_indexed(path) for path in normalized}
    except Exception:
        status = {}
    return {
        "backend": backend,
        "model": model,
        "status": status,
    }


def clear_video_index(
    source_file: str,
    *,
    backend: str | None = None,
    model: str | None = None,
) -> dict:
    """Clear index data for one video file in the resolved backend/model."""
    abs_path = os.path.abspath(source_file)
    backend, model = resolve_index_backend_model(backend, model)
    store = SentryStore(backend=backend, model=model)
    removed_chunks = store.remove_file(abs_path)
    return {
        "backend": backend,
        "model": model,
        "source_file": abs_path,
        "removed_chunks": removed_chunks,
    }


def run_index(
    directory: str,
    *,
    chunk_duration: int = 30,
    overlap: int = 5,
    preprocess: bool = True,
    target_resolution: int = 480,
    target_fps: int = 5,
    skip_still: bool = True,
    backend: str | None = None,
    model: str | None = None,
    quantize: bool | None = None,
    force_reindex: bool = False,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Index videos from a file or directory."""
    backend, model = resolve_index_backend_model(backend, model)
    _notify(progress_callback, phase="starting", backend=backend, model=model)

    get_embedder(backend, model=model, quantize=quantize)
    if os.path.isfile(directory):
        videos = [os.path.abspath(directory)]
    else:
        videos = scan_directory(directory)

    if not videos:
        return {
            "indexed_chunks": 0,
            "indexed_files": 0,
            "skipped_chunks": 0,
            "total_chunks": 0,
            "unique_source_files": 0,
            "source_files": [],
            "backend": backend,
            "model": model,
            "supported_extensions": SUPPORTED_VIDEO_EXTENSIONS,
        }

    store = SentryStore(backend=backend, model=model)
    total_files = len(videos)
    indexed_files = 0
    indexed_chunks = 0
    skipped_chunks = 0
    rebuilt_files = 0
    removed_chunks = 0
    processed_files = 0
    processed_chunks = 0
    total_chunks_estimate = 0

    for file_idx, video_path in enumerate(videos, 1):
        abs_path = os.path.abspath(video_path)
        basename = os.path.basename(video_path)

        if store.is_indexed(abs_path):
            if force_reindex:
                removed = store.remove_file(abs_path)
                removed_chunks += removed
                rebuilt_files += 1
                _notify(
                    progress_callback,
                    phase="indexing",
                    current_file=basename,
                    file_index=file_idx,
                    total_files=total_files,
                    processed_files=processed_files,
                    processed_chunks=processed_chunks,
                    total_chunks_estimate=total_chunks_estimate,
                    reindexing=True,
                    removed_chunks=removed,
                )
            else:
                processed_files += 1
                _notify(
                    progress_callback,
                    phase="indexing",
                    current_file=basename,
                    file_index=file_idx,
                    total_files=total_files,
                    processed_files=processed_files,
                    processed_chunks=processed_chunks,
                    total_chunks_estimate=total_chunks_estimate,
                    skipped=True,
                )
                continue

        chunks = chunk_video(abs_path, chunk_duration=chunk_duration, overlap=overlap)
        total_chunks_estimate += len(chunks)
        embedded = []
        files_to_cleanup = []

        for chunk_idx, chunk in enumerate(chunks, 1):
            file_progress_percent = int((chunk_idx / len(chunks)) * 100)
            _notify(
                progress_callback,
                phase="indexing",
                current_file=basename,
                file_index=file_idx,
                total_files=total_files,
                current_chunk=chunk_idx,
                total_chunks_in_file=len(chunks),
                file_progress_percent=file_progress_percent,
                processed_files=processed_files,
                processed_chunks=processed_chunks,
                total_chunks_estimate=total_chunks_estimate,
            )
            if skip_still and is_still_frame_chunk(chunk["chunk_path"], verbose=verbose):
                skipped_chunks += 1
                processed_chunks += 1
                files_to_cleanup.append(chunk["chunk_path"])
                continue

            embed_path = chunk["chunk_path"]
            if preprocess:
                embed_path = preprocess_chunk(
                    embed_path,
                    target_resolution=target_resolution,
                    target_fps=target_fps,
                )
                if embed_path != chunk["chunk_path"]:
                    files_to_cleanup.append(embed_path)

            embedding = get_embedder().embed_video_chunk(embed_path, verbose=verbose)
            embedded.append({**chunk, "embedding": embedding})
            processed_chunks += 1
            files_to_cleanup.append(chunk["chunk_path"])

        for f in files_to_cleanup:
            try:
                os.unlink(f)
            except OSError:
                pass
        if chunks:
            tmp_dir = os.path.dirname(chunks[0]["chunk_path"])
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if embedded:
            store.add_chunks(embedded)
            indexed_files += 1
            indexed_chunks += len(embedded)

        processed_files += 1

    stats = store.get_stats()
    _notify(
        progress_callback,
        phase="completed",
        processed_files=processed_files,
        total_files=total_files,
        processed_chunks=processed_chunks,
        total_chunks_estimate=total_chunks_estimate,
    )
    return {
        "indexed_chunks": indexed_chunks,
        "indexed_files": indexed_files,
        "skipped_chunks": skipped_chunks,
        "rebuilt_files": rebuilt_files,
        "removed_chunks": removed_chunks,
        "total_chunks": stats["total_chunks"],
        "unique_source_files": stats["unique_source_files"],
        "source_files": stats["source_files"],
        "backend": backend,
        "model": model,
    }
def get_index_rebuild_candidates(
    directory: str,
    *,
    backend: str | None = None,
    model: str | None = None,
) -> dict:
    """Inspect target videos and return already-indexed files for this backend/model."""
    backend, model = resolve_index_backend_model(backend, model)
    if os.path.isfile(directory):
        videos = [os.path.abspath(directory)]
    else:
        videos = scan_directory(directory)
    store = SentryStore(backend=backend, model=model)
    indexed_files = [v for v in videos if store.is_indexed(os.path.abspath(v))]
    return {
        "backend": backend,
        "model": model,
        "videos": videos,
        "indexed_files": indexed_files,
        "total_videos": len(videos),
        "indexed_count": len(indexed_files),
    }


def run_search(
    query: str,
    *,
    n_results: int = 5,
    recall: int | None = None,
    threshold: float = 0.41,
    backend: str | None = None,
    model: str | None = None,
    quantize: bool | None = None,
    rerank: bool = False,
    verbose: bool = False,
) -> dict:
    """Run query search and return normalized result payload."""
    backend, model = resolve_search_backend_model(backend, model)
    store = SentryStore(backend=backend, model=model)
    requested_results = max(1, n_results)
    candidate_results = recall if recall is not None else requested_results * 5
    candidate_results = max(candidate_results, requested_results)
    if store.get_stats()["total_chunks"] == 0:
        detected_backend, detected_model = detect_index()
        if detected_backend == backend and detected_model and detected_model != model:
            message = (
                f"No footage indexed with model '{model}'. "
                f"Current index uses '{detected_model}'."
            )
        elif detected_backend and detected_backend != backend:
            message = (
                f"No footage indexed with backend '{backend}'. "
                f"Current index uses '{detected_backend}'."
            )
        else:
            message = "No indexed footage found. Run index first."
        return {
                "results": [],
                "backend": backend,
                "model": model,
                "threshold": threshold,
            "best_score": None,
            "low_confidence": False,
            "message": message,
        }

    get_embedder(backend, model=model, quantize=quantize)

    results = search_footage(query, store, n_results=candidate_results, verbose=verbose)

    if rerank and results:
        reranker = get_reranker()
        reranked = reranker.rerank(query, results)
        results = [
            {
                **{k: v for k, v in item.items() if k != "rerank_score"},
                "vector_score": item["similarity_score"],
                "similarity_score": item["rerank_score"],
            }
            for item in reranked[:requested_results]
        ]
    else:
        results = [
            {
                **item,
                "vector_score": item["similarity_score"],
            }
            for item in results[:requested_results]
        ]

    best_score = results[0]["similarity_score"] if results else None
    low_confidence = bool(best_score is not None and best_score < threshold)
    return {
        "results": results,
        "backend": backend,
        "model": model,
        "threshold": threshold,
        "best_score": best_score,
        "low_confidence": low_confidence,
        "message": "" if results else "No results found.",
    }


def run_trim(
    *,
    results: list[dict],
    selected_indices: list[int],
    output_dir: str = str(CLIPS_DIR),
    overlay: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Trim selected search results into clips."""
    if not results:
        raise ValueError("results is empty.")
    if not selected_indices:
        raise ValueError("selected_indices is empty.")

    unique_indices = sorted(set(selected_indices))
    for idx in unique_indices:
        if idx < 0 or idx >= len(results):
            raise ValueError(f"selected index out of range: {idx}")

    output_dir = os.path.expanduser(output_dir)
    resolved_output_dir = resolve_project_path(output_dir)
    ensure_dir(resolved_output_dir)

    clips: list[str] = []
    for item_idx, idx in enumerate(unique_indices, start=1):
        item = results[idx]
        _notify(
            progress_callback,
            phase="trimming",
            current=item_idx,
            total=len(unique_indices),
            source_file=item["source_file"],
        )
        output_name = _safe_filename(item["source_file"], item["start_time"], item["end_time"])
        output_path = os.path.join(str(resolved_output_dir), output_name)
        clip_path = trim_clip(
            source_file=item["source_file"],
            start_time=item["start_time"],
            end_time=item["end_time"],
            output_path=output_path,
        )
        if overlay:
            _apply_overlay_to_clip(
                clip_path,
                source_file=item["source_file"],
                start_time=item["start_time"],
                end_time=item["end_time"],
            )
        clips.append(clip_path)
    _notify(progress_callback, phase="completed", total=len(unique_indices))
    return {"output_dir": str(resolved_output_dir), "clips": clips}


def _notify(progress_callback: ProgressCallback | None, **payload) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _overlay_output_path(path: str) -> str:
    base, _ext = os.path.splitext(path)
    return f"{base}_overlay.mp4"


def _apply_overlay_to_clip(
    clip_path: str,
    *,
    source_file: str,
    start_time: float,
    end_time: float,
    replace: bool = True,
) -> bool:
    samples = get_metadata_samples(source_file, start_time, end_time)
    if samples is None:
        return False

    location = None
    mid = samples[len(samples) // 2]
    lat = mid.get("latitude_deg", 0.0)
    lon = mid.get("longitude_deg", 0.0)
    if lat and lon:
        location = reverse_geocode(lat, lon)

    overlay_path = _overlay_output_path(clip_path)
    result_path = apply_overlay(
        clip_path,
        overlay_path,
        samples,
        location,
        source_file=source_file,
        start_time=start_time,
    )
    if result_path == overlay_path and os.path.isfile(overlay_path):
        if replace:
            os.replace(overlay_path, clip_path)
        return True
    return False
