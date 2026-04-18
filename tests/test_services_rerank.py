"""Tests for reranked search behavior."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sentrysearch.reranker import RerankerError
from sentrysearch.services import run_search


def _fake_result(source_file: str, score: float) -> dict:
    return {
        "source_file": source_file,
        "start_time": 0.0,
        "end_time": 5.0,
        "similarity_score": score,
    }


def test_run_search_without_rerank_preserves_vector_scores(monkeypatch):
    fake_store = MagicMock()
    fake_store.get_stats.return_value = {"total_chunks": 3}

    search_calls = []

    def _fake_search_footage(query, store, n_results=5, verbose=False):
        search_calls.append(n_results)
        return [_fake_result("a.mp4", 0.9), _fake_result("b.mp4", 0.8)]

    monkeypatch.setattr("sentrysearch.services.SentryStore", lambda **kwargs: fake_store)
    monkeypatch.setattr("sentrysearch.services.get_embedder", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("sentrysearch.services.search_footage", _fake_search_footage)

    payload = run_search(
        "query",
        backend="local",
        model="qwen2b",
        n_results=2,
        rerank=False,
    )

    assert search_calls == [10]
    assert [item["source_file"] for item in payload["results"]] == ["a.mp4", "b.mp4"]
    assert payload["results"][0]["vector_score"] == 0.9
    assert payload["results"][0]["similarity_score"] == 0.9


def test_run_search_uses_explicit_recall_and_truncates_results(monkeypatch):
    fake_store = MagicMock()
    fake_store.get_stats.return_value = {"total_chunks": 3}

    search_calls = []

    def _fake_search_footage(query, store, n_results=5, verbose=False):
        search_calls.append(n_results)
        return [
            _fake_result("a.mp4", 0.9),
            _fake_result("b.mp4", 0.8),
            _fake_result("c.mp4", 0.7),
        ]

    monkeypatch.setattr("sentrysearch.services.SentryStore", lambda **kwargs: fake_store)
    monkeypatch.setattr("sentrysearch.services.get_embedder", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("sentrysearch.services.search_footage", _fake_search_footage)

    payload = run_search(
        "query",
        backend="local",
        model="qwen2b",
        n_results=2,
        recall=3,
        rerank=False,
    )

    assert search_calls == [3]
    assert [item["source_file"] for item in payload["results"]] == ["a.mp4", "b.mp4"]
    assert len(payload["results"]) == 2
    assert payload["results"][0]["vector_score"] == 0.9
    assert payload["results"][0]["similarity_score"] == 0.9


def test_run_search_floors_recall_to_requested_results(monkeypatch):
    fake_store = MagicMock()
    fake_store.get_stats.return_value = {"total_chunks": 3}

    search_calls = []

    def _fake_search_footage(query, store, n_results=5, verbose=False):
        search_calls.append(n_results)
        return [_fake_result("a.mp4", 0.9), _fake_result("b.mp4", 0.8)]

    monkeypatch.setattr("sentrysearch.services.SentryStore", lambda **kwargs: fake_store)
    monkeypatch.setattr("sentrysearch.services.get_embedder", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("sentrysearch.services.search_footage", _fake_search_footage)

    payload = run_search(
        "query",
        backend="local",
        model="qwen2b",
        n_results=2,
        recall=1,
        rerank=False,
    )

    assert search_calls == [2]
    assert len(payload["results"]) == 2
    assert [item["source_file"] for item in payload["results"]] == ["a.mp4", "b.mp4"]


def test_run_search_with_rerank_expands_candidates_and_resorts(monkeypatch):
    fake_store = MagicMock()
    fake_store.get_stats.return_value = {"total_chunks": 3}

    search_calls = []

    def _fake_search_footage(query, store, n_results=5, verbose=False):
        search_calls.append(n_results)
        return [
            _fake_result("a.mp4", 0.9),
            _fake_result("b.mp4", 0.8),
            _fake_result("c.mp4", 0.7),
        ]

    class _FakeReranker:
        def rerank(self, query, candidates, instruction=None):
            scores = {"a.mp4": 0.2, "b.mp4": 0.95, "c.mp4": 0.5}
            out = []
            for item in candidates:
                out.append({**item, "rerank_score": scores[item["source_file"]]})
            out.sort(key=lambda r: r["rerank_score"], reverse=True)
            return out

    monkeypatch.setattr("sentrysearch.services.SentryStore", lambda **kwargs: fake_store)
    monkeypatch.setattr("sentrysearch.services.get_embedder", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("sentrysearch.services.search_footage", _fake_search_footage)
    monkeypatch.setattr("sentrysearch.services.get_reranker", lambda: _FakeReranker())

    payload = run_search(
        "query",
        backend="local",
        model="qwen2b",
        n_results=2,
        rerank=True,
    )

    assert search_calls == [10]
    assert [item["source_file"] for item in payload["results"]] == ["b.mp4", "c.mp4"]
    assert payload["results"][0]["similarity_score"] == 0.95
    assert payload["results"][0]["vector_score"] == 0.8
    assert "rerank_score" not in payload["results"][0]


def test_run_search_rerank_error_bubbles(monkeypatch):
    fake_store = MagicMock()
    fake_store.get_stats.return_value = {"total_chunks": 3}

    monkeypatch.setattr("sentrysearch.services.SentryStore", lambda **kwargs: fake_store)
    monkeypatch.setattr("sentrysearch.services.get_embedder", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "sentrysearch.services.search_footage",
        lambda *a, **kw: [_fake_result("a.mp4", 0.9)],
    )
    monkeypatch.setattr(
        "sentrysearch.services.get_reranker",
        lambda: (_ for _ in ()).throw(RerankerError("reranker missing")),
    )

    with pytest.raises(RerankerError, match="reranker missing"):
        run_search(
            "query",
            backend="local",
            model="qwen2b",
            n_results=1,
            rerank=True,
        )


def test_local_reranker_loads_script_without_eval(tmp_path):
    model_dir = tmp_path / "Qwen3-VL-Reranker-2B"
    scripts_dir = model_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "qwen3_vl_reranker.py").write_text(
        """
class Qwen3VLReranker:
    def __init__(self, model_name_or_path: str, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs

    def process(self, inputs):
        return 0.75
""".strip()
        + "\n",
        encoding="utf-8",
    )

    from sentrysearch import reranker as reranker_module

    reranker_module.reset_reranker()
    reranker = reranker_module.LocalReranker(model_path=Path(model_dir))

    score = reranker.score("query", {"image": "frame.jpg"})

    assert score == pytest.approx(0.75)


def test_local_reranker_fallback_on_mm_token_type_index_error():
    import torch

    class _FakeScriptModel:
        def process(self, _inputs):
            raise TypeError("only integer tensors of a single element can be converted to an index")

        def format_mm_instruction(self, *args, **kwargs):
            return {"pair": True}

        def tokenize(self, _pairs):
            return {
                "input_ids": torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                "mm_token_type_ids": [[0, 1]],
            }

        def compute_scores(self, inputs):
            assert isinstance(inputs["mm_token_type_ids"], torch.Tensor)
            assert inputs["mm_token_type_ids"].shape == (1, 4)
            return [0.88]

    from sentrysearch.reranker import LocalReranker

    reranker = LocalReranker(model_path=Path("."))
    reranker._model = _FakeScriptModel()
    reranker._device = "cpu"

    score = reranker.score("query", {"image": "frame.jpg", "text": "sample.mp4"})

    assert score == pytest.approx(0.88)


def test_local_reranker_fallback_handles_batch_encoding_tokens():
    import torch
    from transformers import BatchEncoding

    class _FakeScriptModel:
        def process(self, _inputs):
            raise TypeError("only integer tensors of a single element can be converted to an index")

        def format_mm_instruction(self, *args, **kwargs):
            return {"pair": True}

        def tokenize(self, _pairs):
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                    "mm_token_type_ids": [[0, 1]],
                }
            )

        def compute_scores(self, inputs):
            assert isinstance(inputs["mm_token_type_ids"], torch.Tensor)
            assert inputs["mm_token_type_ids"].shape == (1, 4)
            assert isinstance(inputs["attention_mask"], torch.Tensor)
            return [0.92]

    from sentrysearch.reranker import LocalReranker

    reranker = LocalReranker(model_path=Path("."))
    reranker._model = _FakeScriptModel()
    reranker._device = "cpu"

    score = reranker.score("query", {"image": "frame.jpg", "text": "sample.mp4"})

    assert score == pytest.approx(0.92)
