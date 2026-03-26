"""Tests for sentrysearch.store."""

import math

import pytest

from sentrysearch.store import _make_chunk_id


class TestMakeChunkId:
    def test_deterministic(self):
        id1 = _make_chunk_id("video.mp4", 30.0)
        id2 = _make_chunk_id("video.mp4", 30.0)
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = _make_chunk_id("video.mp4", 30.0)
        id2 = _make_chunk_id("video.mp4", 60.0)
        id3 = _make_chunk_id("other.mp4", 30.0)
        assert id1 != id2
        assert id1 != id3

    def test_returns_hex_string(self):
        cid = _make_chunk_id("test.mp4", 0.0)
        assert len(cid) == 16
        int(cid, 16)  # should not raise


def _make_embedding(seed: float = 1.0, dim: int = 768) -> list[float]:
    vec = [math.sin(seed + i * 0.1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class TestSentryStore:
    def test_empty_store_stats(self, tmp_store):
        stats = tmp_store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["unique_source_files"] == 0
        assert stats["source_files"] == []

    def test_empty_store_search(self, tmp_store):
        results = tmp_store.search(_make_embedding(), n_results=5)
        assert results == []

    def test_add_chunk_and_retrieve(self, tmp_store):
        emb = _make_embedding(seed=1.0)
        tmp_store.add_chunk(
            chunk_id="chunk001",
            embedding=emb,
            metadata={
                "source_file": "/path/to/video.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
            },
        )
        stats = tmp_store.get_stats()
        assert stats["total_chunks"] == 1
        assert stats["unique_source_files"] == 1
        assert "/path/to/video.mp4" in stats["source_files"]

    def test_search_returns_sorted_results(self, tmp_store):
        emb_a = _make_embedding(seed=1.0)
        emb_b = _make_embedding(seed=100.0)
        tmp_store.add_chunk("a", emb_a, {
            "source_file": "vid.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("b", emb_b, {
            "source_file": "vid.mp4", "start_time": 30.0, "end_time": 60.0,
        })
        results = tmp_store.search(emb_a, n_results=2)
        assert len(results) == 2
        assert results[0]["start_time"] == 0.0
        assert results[0]["score"] > results[1]["score"]

    def test_add_chunks_batch(self, tmp_store):
        chunks = [
            {
                "source_file": "batch.mp4",
                "start_time": float(i * 30),
                "end_time": float((i + 1) * 30),
                "embedding": _make_embedding(seed=float(i)),
            }
            for i in range(5)
        ]
        tmp_store.add_chunks(chunks)
        stats = tmp_store.get_stats()
        assert stats["total_chunks"] == 5
        assert stats["unique_source_files"] == 1

    def test_upsert_overwrites(self, tmp_store):
        emb1 = _make_embedding(seed=1.0)
        emb2 = _make_embedding(seed=2.0)
        meta = {"source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0}
        tmp_store.add_chunk("same_id", emb1, meta)
        tmp_store.add_chunk("same_id", emb2, meta)
        assert tmp_store.get_stats()["total_chunks"] == 1

    def test_is_indexed(self, tmp_store):
        assert not tmp_store.is_indexed("nonexistent.mp4")
        tmp_store.add_chunk("x", _make_embedding(), {
            "source_file": "found.mp4", "start_time": 0.0, "end_time": 10.0,
        })
        assert tmp_store.is_indexed("found.mp4")
        assert not tmp_store.is_indexed("other.mp4")

    def test_self_similarity_near_one(self, tmp_store):
        emb = _make_embedding(seed=42.0)
        tmp_store.add_chunk("self", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        results = tmp_store.search(emb, n_results=1)
        assert len(results) == 1
        assert results[0]["score"] > 0.99
