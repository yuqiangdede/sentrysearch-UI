"""Tests for web console app."""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from sentrysearch.webapp import create_app
from sentrysearch.reranker import RerankerError


def _wait_job(client: TestClient, job_id: str, timeout: float = 3.0) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_stats_endpoint_empty(monkeypatch):
    monkeypatch.setattr(
        "sentrysearch.webapp.get_stats",
        lambda: {
            "total_chunks": 0,
            "unique_source_files": 0,
            "source_files": [],
            "backend": "gemini",
            "model": None,
        },
    )
    client = TestClient(create_app())
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 0
    assert data["backend"] == "gemini"


def test_home_page_contains_index_page_content():
    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="idxEngine"' in html
    assert '<option value="qwen8b">Qwen3-VL-Embedding-8B 本地</option>' in html
    assert 'id="idxQuantize" type="checkbox" checked' in html
    assert 'id="searchEngine"' not in html


def test_search_page_contains_search_page_content():
    client = TestClient(create_app())
    resp = client.get("/search")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="searchEngine"' in html
    assert 'id="searchRecall"' in html
    assert 'id="searchRecall" type="number" min="1" value="25"' in html
    assert 'id="searchRerank" type="checkbox" checked' in html
    assert 'id="searchQuantize" type="checkbox" checked' in html
    assert 'searchParams.set("start"' in html
    assert 'searchParams.set("end"' in html
    assert 'id="idxEngine"' not in html


def test_stats_endpoint_with_data(monkeypatch):
    monkeypatch.setattr(
        "sentrysearch.webapp.get_stats",
        lambda: {
            "total_chunks": 12,
            "unique_source_files": 2,
            "source_files": ["/a.mp4", "/b.mp4"],
            "backend": "local",
            "model": "qwen2b",
        },
    )
    client = TestClient(create_app())
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 12
    assert data["model"] == "qwen2b"


def test_upload_video_and_list(monkeypatch, tmp_path):
    monkeypatch.setattr("sentrysearch.webapp.UPLOAD_DIR", Path(tmp_path) / "uploads" / "videos")
    client = TestClient(create_app())

    upload = client.post(
        "/api/upload-video",
        files={"file": ("clip.mp4", b"fake-video", "video/mp4")},
    )
    assert upload.status_code == 200
    payload = upload.json()
    assert payload["name"].endswith(".mp4")
    assert payload["size_bytes"] == len(b"fake-video")

    listing = client.get("/api/uploads")
    assert listing.status_code == 200
    data = listing.json()
    assert len(data["videos"]) == 1
    assert data["videos"][0]["name"].endswith(".mp4")


def test_upload_rejects_invalid_extension(monkeypatch, tmp_path):
    monkeypatch.setattr("sentrysearch.webapp.UPLOAD_DIR", Path(tmp_path) / "uploads" / "videos")
    client = TestClient(create_app())
    resp = client.post(
        "/api/upload-video",
        files={"file": ("bad.txt", b"not-video", "text/plain")},
    )
    assert resp.status_code == 400


def test_search_endpoint_and_validation(monkeypatch):
    calls = []

    def _fake_run_search(**kwargs):
        calls.append(kwargs)
        return {
            "results": [{
                "source_file": "/tmp/a.mp4",
                "start_time": 1.0,
                "end_time": 4.0,
                "similarity_score": 0.9,
                "vector_score": 0.7,
            }],
            "backend": "local",
            "model": "qwen2b",
            "threshold": kwargs["threshold"],
            "best_score": 0.9,
            "low_confidence": False,
            "message": "",
        }

    monkeypatch.setattr(
        "sentrysearch.webapp.run_search",
        _fake_run_search,
    )
    client = TestClient(create_app())

    ok = client.post(
        "/api/search",
        json={"query": "red car", "results": 3, "recall": 15, "threshold": 0.41, "rerank": True},
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert len(payload["results"]) == 1
    assert payload["results"][0]["similarity_score"] == 0.9
    assert payload["results"][0]["preview_url"].startswith("/api/media/")
    assert payload["results"][0]["vector_score"] == 0.7
    assert calls[0]["n_results"] == 3
    assert calls[0]["recall"] == 15
    assert calls[0]["rerank"] is True

    preview = client.get(payload["results"][0]["preview_url"])
    assert preview.status_code == 404  # token exists but source file in test is synthetic path

    bad = client.post("/api/search", json={"query": ""})
    assert bad.status_code == 422


def test_media_endpoint_generates_browser_preview(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-video")
    captured = {}

    def _fake_create_browser_preview_clip(*, source_file, start_time, end_time, output_path, padding=2.0):
        captured.update(
            {
                "source_file": source_file,
                "start_time": start_time,
                "end_time": end_time,
                "output_path": output_path,
                "padding": padding,
            }
        )
        Path(output_path).write_bytes(b"preview-bytes")
        return output_path

    monkeypatch.setattr("sentrysearch.webapp.create_browser_preview_clip", _fake_create_browser_preview_clip)
    monkeypatch.setattr(
        "sentrysearch.webapp.run_search",
        lambda **kwargs: {
            "results": [{
                "source_file": str(source),
                "start_time": 1.0,
                "end_time": 4.0,
                "similarity_score": 0.9,
            }],
            "backend": "local",
            "model": "qwen2b",
            "threshold": kwargs["threshold"],
            "best_score": 0.9,
            "low_confidence": False,
            "message": "",
        },
    )

    client = TestClient(create_app())
    ok = client.post(
        "/api/search",
        json={"query": "red car", "results": 3, "threshold": 0.41, "rerank": True},
    )
    assert ok.status_code == 200
    preview_url = ok.json()["results"][0]["preview_url"]

    media = client.get(f"{preview_url}?start=1.0&end=4.0")
    assert media.status_code == 200
    assert media.content == b"preview-bytes"
    assert captured["source_file"] == str(source)
    assert captured["start_time"] == 1.0
    assert captured["end_time"] == 4.0


def test_search_endpoint_passes_rerank(monkeypatch):
    calls = []

    def _fake_run_search(**kwargs):
        calls.append(kwargs)
        return {
            "results": [],
            "backend": "local",
            "model": "qwen2b",
            "threshold": kwargs["threshold"],
            "best_score": None,
            "low_confidence": False,
            "message": "No results found.",
        }

    monkeypatch.setattr("sentrysearch.webapp.run_search", _fake_run_search)
    client = TestClient(create_app())

    resp = client.post(
        "/api/search",
        json={"query": "red car", "results": 3, "recall": 9, "threshold": 0.41, "rerank": True},
    )
    assert resp.status_code == 200
    assert calls[0]["rerank"] is True
    assert calls[0]["recall"] == 9


def test_search_endpoint_returns_reranker_error(monkeypatch):
    def _raise(**kwargs):
        raise RerankerError("missing reranker model")

    monkeypatch.setattr("sentrysearch.webapp.run_search", _raise)
    client = TestClient(create_app())

    resp = client.post(
        "/api/search",
        json={"query": "red car", "results": 3, "threshold": 0.41, "rerank": True},
    )
    assert resp.status_code == 503
    assert "missing reranker model" in resp.json()["detail"]


def test_index_job_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sentrysearch.webapp.get_index_rebuild_candidates",
        lambda **kwargs: {
            "backend": "local",
            "model": "qwen2b",
            "videos": [str(tmp_path / "a.mp4")],
            "indexed_files": [],
            "total_videos": 1,
            "indexed_count": 0,
        },
    )

    def _fake_run_index(**kwargs):
        kwargs["progress_callback"]({"phase": "indexing", "processed_files": 1, "total_files": 1})
        kwargs["progress_callback"]({"phase": "completed", "processed_files": 1, "total_files": 1})
        return {
            "indexed_chunks": 4,
            "indexed_files": 1,
            "skipped_chunks": 0,
            "rebuilt_files": 0,
            "removed_chunks": 0,
            "total_chunks": 4,
            "unique_source_files": 1,
            "source_files": [str(tmp_path / "a.mp4")],
            "backend": "local",
            "model": "qwen2b",
        }

    monkeypatch.setattr("sentrysearch.webapp.run_index", _fake_run_index)
    client = TestClient(create_app())

    resp = client.post("/api/index", json={"directory": str(tmp_path)})
    assert resp.status_code == 200
    job = _wait_job(client, resp.json()["job_id"])
    assert job["status"] == "succeeded"
    assert job["result"]["indexed_chunks"] == 4
    assert job["progress"]["phase"] == "completed"


def test_index_returns_conflict_when_already_indexed(monkeypatch, tmp_path):
    target = tmp_path / "a.mp4"
    target.write_bytes(b"")
    monkeypatch.setattr(
        "sentrysearch.webapp.get_index_rebuild_candidates",
        lambda **kwargs: {
            "backend": "local",
            "model": "qwen2b",
            "videos": [str(target)],
            "indexed_files": [str(target)],
            "total_videos": 1,
            "indexed_count": 1,
        },
    )
    client = TestClient(create_app())
    resp = client.post("/api/index", json={"directory": str(target)})
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "indexed_exists"
    assert body["indexed_count"] == 1


def test_index_force_reindex_runs_job(monkeypatch, tmp_path):
    target = tmp_path / "a.mp4"
    target.write_bytes(b"")
    monkeypatch.setattr(
        "sentrysearch.webapp.get_index_rebuild_candidates",
        lambda **kwargs: {
            "backend": "local",
            "model": "qwen2b",
            "videos": [str(target)],
            "indexed_files": [str(target)],
            "total_videos": 1,
            "indexed_count": 1,
        },
    )
    calls = []

    def _fake_run_index(**kwargs):
        calls.append(kwargs.get("force_reindex"))
        kwargs["progress_callback"]({"phase": "completed"})
        return {
            "indexed_chunks": 1,
            "indexed_files": 1,
            "skipped_chunks": 0,
            "rebuilt_files": 1,
            "removed_chunks": 2,
            "total_chunks": 1,
            "unique_source_files": 1,
            "source_files": [str(tmp_path / "a.mp4")],
            "backend": "local",
            "model": "qwen2b",
        }

    monkeypatch.setattr("sentrysearch.webapp.run_index", _fake_run_index)
    client = TestClient(create_app())
    resp = client.post(
        "/api/index",
        json={"directory": str(target), "force_reindex": True},
    )
    assert resp.status_code == 200
    job = _wait_job(client, resp.json()["job_id"])
    assert job["status"] == "succeeded"
    assert calls == [True]


def test_index_directory_with_partial_existing_should_not_conflict(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sentrysearch.webapp.get_index_rebuild_candidates",
        lambda **kwargs: {
            "backend": "local",
            "model": "qwen2b",
            "videos": [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
            "indexed_files": [str(tmp_path / "a.mp4")],
            "total_videos": 2,
            "indexed_count": 1,
        },
    )

    def _fake_run_index(**kwargs):
        kwargs["progress_callback"]({"phase": "completed"})
        return {
            "indexed_chunks": 1,
            "indexed_files": 1,
            "skipped_chunks": 0,
            "rebuilt_files": 0,
            "removed_chunks": 0,
            "total_chunks": 2,
            "unique_source_files": 2,
            "source_files": [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
            "backend": "local",
            "model": "qwen2b",
        }

    monkeypatch.setattr("sentrysearch.webapp.run_index", _fake_run_index)
    client = TestClient(create_app())
    resp = client.post("/api/index", json={"directory": str(tmp_path)})
    assert resp.status_code == 200


def test_index_directory_all_existing_conflicts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sentrysearch.webapp.get_index_rebuild_candidates",
        lambda **kwargs: {
            "backend": "local",
            "model": "qwen2b",
            "videos": [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
            "indexed_files": [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
            "total_videos": 2,
            "indexed_count": 2,
        },
    )
    client = TestClient(create_app())
    resp = client.post("/api/index", json={"directory": str(tmp_path)})
    assert resp.status_code == 409
    payload = resp.json()
    assert payload["code"] == "indexed_exists"
    assert payload["indexed_count"] == 2


def test_index_status_endpoint(monkeypatch):
    monkeypatch.setattr(
        "sentrysearch.webapp.get_video_index_status",
        lambda **kwargs: {
            "backend": "local",
            "model": "qwen2b",
            "status": {
                "/tmp/a.mp4": True,
                "/tmp/b.mp4": False,
            },
        },
    )
    client = TestClient(create_app())
    resp = client.post(
        "/api/index-status",
        json={
            "source_files": ["/tmp/a.mp4", "/tmp/b.mp4"],
            "backend": "local",
            "model": "qwen2b",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"]["/tmp/a.mp4"] is True
    assert payload["status"]["/tmp/b.mp4"] is False


def test_clear_video_index_endpoint(monkeypatch):
    monkeypatch.setattr(
        "sentrysearch.webapp.clear_video_index",
        lambda **kwargs: {
            "backend": "local",
            "model": "qwen8b",
            "source_file": kwargs["source_file"],
            "removed_chunks": 3,
        },
    )
    client = TestClient(create_app())
    resp = client.post(
        "/api/index/clear",
        json={
            "source_file": "/tmp/a.mp4",
            "backend": "local",
            "model": "qwen8b",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["removed_chunks"] == 3
    assert payload["source_file"] == "/tmp/a.mp4"


def test_trim_job_lifecycle_and_download(monkeypatch, tmp_path):
    clip = tmp_path / "clip1.mp4"
    clip.write_bytes(b"test")

    def _fake_run_trim(**kwargs):
        kwargs["progress_callback"]({"phase": "trimming", "current": 1, "total": 1})
        kwargs["progress_callback"]({"phase": "completed", "total": 1})
        return {"output_dir": str(tmp_path), "clips": [str(clip)]}

    monkeypatch.setattr("sentrysearch.webapp.run_trim", _fake_run_trim)
    client = TestClient(create_app())

    resp = client.post(
        "/api/trim",
        json={
            "results": [{
                "source_file": str(tmp_path / "src.mp4"),
                "start_time": 0.0,
                "end_time": 2.0,
                "similarity_score": 0.95,
            }],
            "selected_indices": [0],
            "output_dir": str(tmp_path),
            "overlay": False,
        },
    )
    assert resp.status_code == 200
    job = _wait_job(client, resp.json()["job_id"])
    assert job["status"] == "succeeded"
    assert len(job["result"]["clips"]) == 1
    download_url = job["result"]["clips"][0]["download_url"]
    media_url = job["result"]["clips"][0]["preview_url"]

    dl = client.get(download_url)
    assert dl.status_code == 200
    assert dl.content == b"test"
    media = client.get(media_url)
    assert media.status_code == 200


def test_multiple_jobs_isolated(monkeypatch, tmp_path):
    def _fake_run_index(**kwargs):
        directory = kwargs["directory"]
        kwargs["progress_callback"]({"phase": "indexing", "current_file": directory})
        return {
            "indexed_chunks": 1,
            "indexed_files": 1,
            "skipped_chunks": 0,
            "total_chunks": 1,
            "unique_source_files": 1,
            "source_files": [directory],
            "backend": "gemini",
            "model": None,
        }

    monkeypatch.setattr("sentrysearch.webapp.run_index", _fake_run_index)
    client = TestClient(create_app())

    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()

    r1 = client.post("/api/index", json={"directory": str(d1)})
    r2 = client.post("/api/index", json={"directory": str(d2)})
    j1 = _wait_job(client, r1.json()["job_id"])
    j2 = _wait_job(client, r2.json()["job_id"])

    assert j1["status"] == "succeeded"
    assert j2["status"] == "succeeded"
    assert j1["result"]["source_files"] == [str(d1)]
    assert j2["result"]["source_files"] == [str(d2)]
