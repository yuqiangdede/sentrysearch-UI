"""Local multimodal reranking for search results."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .chunker import _get_ffmpeg_executable
from .paths import TEMP_DIR, ensure_dir, resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RERANKER_DIR = PROJECT_ROOT / "models" / "Qwen3-VL-Reranker-2B"
DEFAULT_INSTRUCTION = "Retrieve videos relevant to the user's query."


class RerankerError(RuntimeError):
    """Raised when the local reranker cannot be loaded or executed."""


_current_reranker: "LocalReranker | None" = None


def get_reranker(model_path: str | Path | None = None) -> "LocalReranker":
    """Return the cached reranker instance."""
    global _current_reranker
    resolved = _resolve_model_path(model_path)
    if _current_reranker is None or _current_reranker.model_path != resolved:
        _current_reranker = LocalReranker(model_path=resolved)
    return _current_reranker


def reset_reranker() -> None:
    """Drop the cached reranker instance."""
    global _current_reranker
    _current_reranker = None


def _resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path is None:
        return DEFAULT_RERANKER_DIR
    candidate = resolve_project_path(model_path)
    return candidate


def _score_to_float(score: Any) -> float:
    """Convert common reranker outputs to a 0..1 relevance score."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch unavailable in minimal envs
        torch = None

    if torch is not None and isinstance(score, torch.Tensor):
        score = score.detach().float().cpu()
        if score.numel() == 1:
            value = float(score.item())
            return value if 0.0 <= value <= 1.0 else float(torch.sigmoid(score).item())
        if score.ndim == 1 and score.numel() == 2:
            probs = torch.softmax(score, dim=-1)
            return float(probs[-1].item())
        if score.ndim > 1 and score.shape[-1] == 2:
            probs = torch.softmax(score, dim=-1)
            return float(probs.reshape(-1, 2)[0, 1].item())
        flat = score.reshape(-1)
        value = float(flat[0].item())
        return value if 0.0 <= value <= 1.0 else float(torch.sigmoid(flat[0]).item())

    if isinstance(score, (list, tuple)):
        if len(score) == 1:
            return _score_to_float(score[0])
        if len(score) == 2 and all(isinstance(x, (int, float)) for x in score):
            a = float(score[0])
            b = float(score[1])
            m = max(a, b)
            exps = [math.exp(a - m), math.exp(b - m)]
            total = sum(exps)
            return exps[1] / total if total else 0.0
        if score:
            return _score_to_float(score[0])

    value = float(score)
    if 0.0 <= value <= 1.0:
        return value
    if value > 60:
        return 1.0
    if value < -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _call_eval_if_available(model: Any) -> None:
    """Call ``eval()`` only for objects that actually implement it."""
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()


def _is_mm_token_type_index_error(exc: Exception) -> bool:
    text = str(exc)
    return "only integer tensors of a single element can be converted to an index" in text


def _import_qwen3_vl_reranker(model_path: Path):
    script_candidates = [
        model_path / "scripts" / "qwen3_vl_reranker.py",
        model_path / "qwen3_vl_reranker.py",
    ]
    for script_path in script_candidates:
        if not script_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "qwen3_vl_reranker_local",
            script_path,
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "Qwen3VLReranker"):
            return module.Qwen3VLReranker
    return None


class LocalReranker:
    """Qwen3-VL reranker wrapper for local search refinement."""

    def __init__(self, model_path: str | Path = DEFAULT_RERANKER_DIR):
        self.model_path = Path(model_path)
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None
        self._loader_kind = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise RerankerError(
                f"Reranker model not found: {self.model_path}. "
                "Copy Qwen3-VL-Reranker-2B into the project models directory first."
            )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise RerankerError(
                "Missing dependencies for local reranking: torch is not installed."
            ) from exc

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        self._device = device
        self._dtype = dtype

        qwen_cls = _import_qwen3_vl_reranker(self.model_path)
        if qwen_cls is not None:
            self._loader_kind = "local_script"
            try:
                self._model = qwen_cls(
                    model_name_or_path=str(self.model_path),
                    torch_dtype=dtype if device != "cpu" else None,
                    attn_implementation="sdpa",
                )
                _call_eval_if_available(self._model)
                return
            except TypeError:
                self._model = qwen_cls(model_name_or_path=str(self.model_path))
                _call_eval_if_available(self._model)
                return
            except Exception as exc:
                raise RerankerError(
                    f"Failed to load reranker from local script at {self.model_path}: {exc}"
                ) from exc

        try:
            from transformers import AutoModelForSequenceClassification, AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise RerankerError(
                "Missing dependencies for local reranking: transformers is not installed."
            ) from exc

        try:
            self._processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=dtype if device != "cpu" else torch.float32,
            )
            if device != "cpu":
                self._model = self._model.to(device)
            _call_eval_if_available(self._model)
            self._loader_kind = "auto"
        except Exception as exc:
            raise RerankerError(
                f"Failed to load reranker model from {self.model_path}: {exc}"
            ) from exc

    def _extract_mid_frame(self, source_file: str, start_time: float, end_time: float, tmpdir: Path) -> str:
        """Extract one representative frame for reranking."""
        ffmpeg_exe = _get_ffmpeg_executable()
        midpoint = max(0.0, (float(start_time) + float(end_time)) / 2.0)
        output_path = tmpdir / "frame.jpg"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss",
            str(midpoint),
            "-i",
            source_file,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not output_path.is_file():
            raise RerankerError(
                f"Failed to extract rerank frame from {source_file}:\n{result.stderr}"
            )
        return str(output_path)

    def _score_with_auto_loader(
        self,
        query: str,
        document: dict[str, Any],
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> float:
        import torch

        query_payload = {"text": query}
        documents = [document]

        if hasattr(self._model, "process"):
            inputs = {
                "instruction": instruction,
                "query": query_payload,
                "documents": documents,
                "fps": 1.0,
            }
            if document.get("image") or document.get("video"):
                inputs["max_frames"] = 1
            try:
                score = self._model.process(inputs)
                return _score_to_float(score)
            except TypeError as exc:
                if _is_mm_token_type_index_error(exc):
                    return self._score_with_local_script_fallback(
                        query=query,
                        document=document,
                        instruction=instruction,
                    )
                raise RerankerError(
                    f"Failed to score candidate with local script reranker: {exc}"
                ) from exc
            except Exception as exc:
                raise RerankerError(
                    f"Failed to score candidate with local script reranker: {exc}"
                ) from exc

        if self._processor is None:
            raise RerankerError("Reranker processor is not available.")

        # Fallback for generic sequence-classification style checkpoints.
        content = [{"type": "text", "text": instruction}]
        content.append({"type": "text", "text": query})
        if "image" in document:
            content.append({"type": "image", "image": document["image"]})
        elif "text" in document:
            content.append({"type": "text", "text": document["text"]})
        else:
            raise RerankerError("Unsupported document payload for reranking.")

        messages = [{"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt", "padding": True}
        if "image" in document:
            kwargs["images"] = [document["image"]]
        inputs = self._processor(**kwargs)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = getattr(outputs, "logits", outputs)
        return _score_to_float(logits)

    def _score_with_local_script_fallback(
        self,
        query: str,
        document: dict[str, Any],
        instruction: str,
    ) -> float:
        import torch

        if not all(hasattr(self._model, name) for name in ("format_mm_instruction", "tokenize", "compute_scores")):
            raise RerankerError(
                "Local reranker script fallback is unavailable: required methods are missing."
            )

        pair = self._model.format_mm_instruction(
            query,
            None,
            None,
            document.get("text"),
            document.get("image"),
            document.get("video"),
            instruction=instruction,
            fps=1.0,
            max_frames=1 if document.get("image") or document.get("video") else None,
        )
        tokenized = self._model.tokenize([pair])

        if not isinstance(tokenized, dict):
            tokenized = dict(tokenized)

        tokenized = self._normalize_mm_token_type_ids(tokenized)
        tokenized = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in tokenized.items()
        }

        with torch.no_grad():
            scores = self._model.compute_scores(tokenized)
        return _score_to_float(scores)

    def _normalize_mm_token_type_ids(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch

        mm_key = "mm_token_type_ids"
        if mm_key not in inputs:
            return inputs

        mm_value = inputs[mm_key]
        if isinstance(mm_value, torch.Tensor):
            return inputs
        if not isinstance(mm_value, (list, tuple)) or not mm_value:
            return inputs

        input_ids = inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            target_len = int(input_ids.shape[-1])
            pad_dtype = input_ids.dtype
            pad_device = input_ids.device
        elif isinstance(input_ids, (list, tuple)) and input_ids:
            first_row = input_ids[0]
            if isinstance(first_row, (list, tuple)):
                target_len = max(len(row) for row in input_ids if isinstance(row, (list, tuple)))
            else:
                target_len = len(input_ids)
            pad_dtype = torch.long
            pad_device = None
        else:
            target_len = max(len(row) for row in mm_value if isinstance(row, (list, tuple)))
            pad_dtype = torch.long
            pad_device = None

        # Qwen processors generally use left padding; use 0 (text token type) for padded slots.
        padded: list[list[int]] = []
        for row in mm_value:
            if not isinstance(row, (list, tuple)):
                return inputs
            if len(row) >= target_len:
                padded.append([int(x) for x in row[-target_len:]])
                continue
            pad = [0] * (target_len - len(row))
            padded.append(pad + [int(x) for x in row])

        tensor = torch.tensor(padded, dtype=pad_dtype)
        if pad_device is not None:
            tensor = tensor.to(pad_device)
        inputs[mm_key] = tensor
        return inputs

    def score(self, query: str, document: dict[str, Any]) -> float:
        self._load_model()
        return self._score_with_auto_loader(query, document)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> list[dict[str, Any]]:
        """Return candidates sorted by reranker score descending."""
        self._load_model()
        scored: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="sentrysearch_rerank_", dir=ensure_dir(TEMP_DIR)) as tmp:
            tmpdir = Path(tmp)
            for item in candidates:
                frame_path = self._extract_mid_frame(
                    item["source_file"],
                    item["start_time"],
                    item["end_time"],
                    tmpdir,
                )
                document = {"image": frame_path}
                if item.get("source_file"):
                    document["text"] = os.path.basename(item["source_file"])
                score = self._score_with_auto_loader(query, document, instruction=instruction)
                scored.append({**item, "rerank_score": score})
        scored.sort(key=lambda r: r["rerank_score"], reverse=True)
        return scored
