"""Tests for sentrysearch.local_embedder (mocked - no torch required)."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.local_embedder import (
    LOCAL_QWEN2B_DIR,
    MODEL_ALIASES,
    LocalEmbedder,
    LocalModelError,
    _ensure_qwen_video_reader_backend,
    detect_default_model,
    normalize_model_key,
)


class TestModelAliases:
    def test_qwen8b_alias_resolves(self):
        embedder = LocalEmbedder(model_name="qwen8b")
        assert embedder._model_name == "Qwen/Qwen3-VL-Embedding-8B"

    def test_qwen2b_alias_resolves(self):
        embedder = LocalEmbedder(model_name="qwen2b")
        assert embedder._model_name == str(LOCAL_QWEN2B_DIR)

    def test_full_hf_id_passed_through(self):
        embedder = LocalEmbedder(model_name="Qwen/Qwen3-VL-Embedding-8B")
        assert embedder._model_name == "Qwen/Qwen3-VL-Embedding-8B"

    def test_custom_model_name_passed_through(self):
        embedder = LocalEmbedder(model_name="custom/my-model")
        assert embedder._model_name == "custom/my-model"

    def test_aliases_dict_has_expected_keys(self):
        assert "qwen8b" in MODEL_ALIASES
        assert "qwen2b" in MODEL_ALIASES


class TestLocalModelError:
    def test_is_runtime_error(self):
        assert issubclass(LocalModelError, RuntimeError)

    def test_message(self):
        err = LocalModelError("missing torch")
        assert "missing torch" in str(err)


class TestLocalEmbedderConstruction:
    def test_default_params(self):
        embedder = LocalEmbedder()
        assert embedder._model_name == str(LOCAL_QWEN2B_DIR)
        assert embedder._dimensions == 768
        assert embedder._model is None

    def test_custom_params(self):
        embedder = LocalEmbedder(model_name="custom/model", dimensions=512)
        assert embedder._model_name == "custom/model"
        assert embedder._dimensions == 512

    def test_dimensions_method(self):
        embedder = LocalEmbedder(dimensions=1024)
        assert embedder.dimensions() == 1024

    def test_quantize_none_by_default(self):
        embedder = LocalEmbedder()
        assert embedder._quantize is None

    def test_quantize_true(self):
        embedder = LocalEmbedder(quantize=True)
        assert embedder._quantize is True

    def test_quantize_false(self):
        embedder = LocalEmbedder(quantize=False)
        assert embedder._quantize is False


class TestLocalEmbedderLoadModel:
    def test_missing_torch_raises_local_model_error(self):
        embedder = LocalEmbedder()
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(LocalModelError, match="Missing dependencies"):
                embedder._load_model()

    def test_load_model_called_once(self):
        embedder = LocalEmbedder()
        embedder._model = MagicMock()  # pretend already loaded
        embedder._load_model()


class TestNormalizeModelKey:
    def test_alias_returned_as_is(self):
        assert normalize_model_key("qwen8b") == "qwen8b"
        assert normalize_model_key("qwen2b") == "qwen2b"

    def test_full_hf_id_reversed_to_alias(self):
        assert normalize_model_key("Qwen/Qwen3-VL-Embedding-8B") == "qwen8b"
        assert normalize_model_key("Qwen/Qwen3-VL-Embedding-2B") == "qwen2b"

    def test_custom_model_sanitized(self):
        assert normalize_model_key("org/my-custom-model") == "org_my_custom_model"


class TestDetectDefaultModel:
    def test_no_torch_returns_qwen2b(self):
        with patch.dict("sys.modules", {"torch": None}):
            result = detect_default_model()
            assert result == "qwen2b"

    def test_cuda_returns_qwen2b(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = detect_default_model()
            assert result == "qwen2b"

    def test_cpu_only_returns_qwen2b(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = detect_default_model()
            assert result == "qwen2b"


class TestVideoReaderBackend:
    def test_prefers_decord_when_available(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "sentrysearch.local_embedder.importlib.util.find_spec",
            return_value=object(),
        ):
            backend = _ensure_qwen_video_reader_backend()
            assert backend == "decord"
            assert os.environ["FORCE_QWENVL_VIDEO_READER"] == "decord"

    def test_preserves_existing_setting(self):
        with patch.dict(os.environ, {"FORCE_QWENVL_VIDEO_READER": "torchvision"}, clear=True), patch(
            "sentrysearch.local_embedder.importlib.util.find_spec",
            return_value=object(),
        ):
            backend = _ensure_qwen_video_reader_backend()
            assert backend == "torchvision"

    def test_falls_back_to_torchvision_when_decord_missing(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "sentrysearch.local_embedder.importlib.util.find_spec",
            return_value=None,
        ):
            backend = _ensure_qwen_video_reader_backend()
            assert backend == "torchvision"
            assert os.environ["FORCE_QWENVL_VIDEO_READER"] == "torchvision"


class TestLocalEmbedderMethods:
    def test_embed_query_calls_load_model(self):
        embedder = LocalEmbedder()
        embedder._load_model = MagicMock(
            side_effect=LocalModelError("no torch in CI")
        )
        with pytest.raises(LocalModelError):
            embedder.embed_query("test query")
        embedder._load_model.assert_called_once()

    def test_embed_video_chunk_calls_load_model(self):
        embedder = LocalEmbedder()
        embedder._load_model = MagicMock(
            side_effect=LocalModelError("no torch in CI")
        )
        with pytest.raises(LocalModelError):
            embedder.embed_video_chunk("/fake/path.mp4")
        embedder._load_model.assert_called_once()
