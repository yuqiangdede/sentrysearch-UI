"""Local video/text embedding using Qwen3-VL-Embedding.

Uses the Qwen3VLForEmbedding model class adapted from the official
Qwen3-VL-Embedding repository (Apache 2.0 license).
Source: https://github.com/QwenLM/Qwen3-VL-Embedding
"""

import os
import sys
import time
import importlib.util
from pathlib import Path

from .base_embedder import BaseEmbedder
from .paths import HF_CACHE_DIR, MODELS_DIR, PROJECT_ROOT

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))


class LocalModelError(RuntimeError):
    """Raised when the local model fails to load or run."""


# Short aliases → full HuggingFace model IDs
MODEL_ALIASES: dict[str, str] = {
    "qwen8b": "Qwen/Qwen3-VL-Embedding-8B",
    "qwen2b": "Qwen/Qwen3-VL-Embedding-2B",
}

# Reverse lookup: full HuggingFace ID → short alias
_REVERSE_ALIASES: dict[str, str] = {v: k for k, v in MODEL_ALIASES.items()}
LOCAL_QWEN2B_DIR = MODELS_DIR / "Qwen3-VL-Embedding-2B"
LOCAL_QWEN8B_DIR = MODELS_DIR / "Qwen3-VL-Embedding-8B"


def _is_qwen2b_reference(model: str) -> bool:
    """Return True when *model* points at the 2B Qwen3-VL embedding model."""
    candidate = Path(model)
    return candidate.name == "Qwen3-VL-Embedding-2B"


def normalize_model_key(model: str) -> str:
    """Return canonical short key for a model (used in collection names)."""
    if model in MODEL_ALIASES:
        return model
    if model in _REVERSE_ALIASES:
        return _REVERSE_ALIASES[model]
    if _is_qwen2b_reference(model):
        return "qwen2b"
    # Custom model: sanitize for use as collection name suffix
    return model.replace("/", "_").replace("-", "_").lower()


def detect_default_model() -> str:
    """Use the project-local 2B model as the default local backend."""
    return "qwen2b"


def _ensure_qwen_video_reader_backend() -> str | None:
    """Select a qwen_vl_utils video reader backend that is known to work.

    qwen_vl_utils prefers torchcodec when it is importable. On Windows that
    can produce noisy DLL-load failures when torchcodec is installed but its
    native dependencies are missing. We therefore pin the backend to decord
    when available and otherwise fall back to torchvision explicitly.
    """
    forced = os.environ.get("FORCE_QWENVL_VIDEO_READER")
    if forced:
        return forced
    if importlib.util.find_spec("decord") is not None:
        os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
        return "decord"
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
    return "torchvision"


class LocalEmbedder(BaseEmbedder):
    """Qwen3-VL-Embedding backend (local GPU inference)."""

    def __init__(
        self,
        model_name: str = "qwen2b",
        dimensions: int = 768,
        quantize: bool | None = None,
    ):
        self._model_name = self._public_model_name(model_name)
        self._model_ref = self._resolve_model_reference(model_name)
        self._dimensions = dimensions
        self._quantize = quantize  # None = auto-detect
        self._model = None
        self._processor = None

    @staticmethod
    def _public_model_name(model_name: str) -> str:
        """Return the user-visible model name used by tests and logs."""
        canonical = normalize_model_key(model_name)
        if canonical == "qwen2b" and LOCAL_QWEN2B_DIR.exists():
            return str(LOCAL_QWEN2B_DIR.resolve())
        if canonical == "qwen8b":
            return MODEL_ALIASES["qwen8b"]
        return model_name

    @staticmethod
    def _resolve_model_reference(model_name: str) -> str:
        """Resolve aliases to a local project path when possible."""
        canonical = normalize_model_key(model_name)
        if canonical == "qwen2b":
            if LOCAL_QWEN2B_DIR.exists():
                return str(LOCAL_QWEN2B_DIR.resolve())
            return model_name
        if canonical == "qwen8b":
            if LOCAL_QWEN8B_DIR.exists():
                return str(LOCAL_QWEN8B_DIR.resolve())
            return model_name

        candidate = Path(model_name)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.exists():
            return str(candidate)
        return model_name

    def _load_model(self):
        if self._model is not None:
            return

        try:
            import torch
            import torch.nn.functional as F  # noqa: F401
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLPreTrainedModel,
                Qwen3VLModel,
                Qwen3VLConfig,
            )
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
            from transformers.cache_utils import Cache
            from transformers.utils import TransformersKwargs
            from transformers.processing_utils import Unpack
        except ImportError as e:
            raise LocalModelError(
                f"Missing dependencies for local backend: {e}\n\n"
                "Install with: uv tool install \".[local]\"\n"
                "For 4-bit quantization: uv tool install \".[local-quantized]\""
            ) from e

        # Prefer the project-local model directory when available.
        print(f"Loading local model from {self._model_ref}...", file=sys.stderr)

        # Determine device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
            print(
                "Warning: No GPU detected, local inference will be very slow.",
                file=sys.stderr,
            )

        # 4-bit quantization: explicit flag or auto-detect
        quantization_config = None
        want_quantize = self._quantize
        if want_quantize is None and device == "cuda":
            # Auto: only quantize when VRAM is tight for the chosen model
            props = torch.cuda.get_device_properties(0)
            # Attribute renamed total_mem → total_memory in recent PyTorch
            total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            vram_gb = total_mem / (1024 ** 3)
            # 8B needs ~16 GB in bf16, 2B needs ~4 GB — add headroom
            needs_gb = 18 if "8B" in self._model_name else 6
            want_quantize = vram_gb < needs_gb
        if want_quantize and device == "cuda":
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                print("Using 4-bit quantization (bitsandbytes)", file=sys.stderr)
            except ImportError:
                if self._quantize is True:
                    raise LocalModelError(
                        "4-bit quantization requested but bitsandbytes is not installed.\n\n"
                        "Install with: uv tool install \".[local-quantized]\""
                    )
        elif want_quantize and device != "cuda":
            if self._quantize is True:
                raise LocalModelError(
                    "4-bit quantization requires CUDA (NVIDIA GPU). "
                    f"Current device: {device}"
                )

        # Build the model class dynamically to avoid top-level imports
        _PreTrained = Qwen3VLPreTrainedModel
        _Model = Qwen3VLModel
        _Config = Qwen3VLConfig
        _Cache = Cache
        _TransformersKwargs = TransformersKwargs
        _Unpack = Unpack

        class _Qwen3VLForEmbedding(_PreTrained):
            """Qwen3-VL wrapper that exposes last_hidden_state for pooling."""
            config: _Config

            def __init__(self, config):
                super().__init__(config)
                self.model = _Model(config)
                self.post_init()

            def get_input_embeddings(self):
                return self.model.get_input_embeddings()

            def set_input_embeddings(self, value):
                self.model.set_input_embeddings(value)

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
                cache_position=None,
                **kwargs,
            ):
                return self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    **kwargs,
                )

        try:
            self._processor = Qwen3VLProcessor.from_pretrained(
                self._model_ref, padding_side="right",
            )

            load_kwargs = dict(trust_remote_code=True)
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            else:
                load_kwargs["torch_dtype"] = dtype

            self._model = _Qwen3VLForEmbedding.from_pretrained(
                self._model_ref, **load_kwargs,
            )
            if quantization_config is None:
                self._model = self._model.to(device)
            self._model.eval()
            print(f"Model loaded on {device}", file=sys.stderr)

        except Exception as e:
            raise LocalModelError(f"Failed to load {self._model_name}: {e}") from e

    @staticmethod
    def _pooling_last(hidden_state, attention_mask):
        """Pool at the last non-padded token position."""
        import torch
        flipped = attention_mask.flip(dims=[1])
        last_pos = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @staticmethod
    def _truncate_and_normalize(embedding, target_dims):
        """MRL dimension truncation: slice first N dims, then L2-normalize."""
        import torch
        import torch.nn.functional as F
        truncated = embedding[:target_dims]
        norm = torch.linalg.norm(truncated)
        if norm > 0:
            truncated = truncated / norm
        return truncated.cpu().float().tolist()

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        self._load_model()

        import torch
        import torch.nn.functional as F
        from pathlib import Path

        _ensure_qwen_video_reader_backend()
        from qwen_vl_utils import process_vision_info

        chunk_path = Path(chunk_path)
        if not chunk_path.exists():
            raise LocalModelError(f"Chunk file not found: {chunk_path}")

        if verbose:
            size_kb = os.path.getsize(chunk_path) / 1024
            print(f"    [verbose] embedding {size_kb:.0f}KB chunk locally", file=sys.stderr)

        t0 = time.monotonic()

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Represent the video for retrieval."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        # Use a plain filesystem path here. decord on Windows
                        # does not handle file:// URIs reliably.
                        "video": str(chunk_path.resolve()),
                        "fps": 1.0,
                        "max_frames": 32,
                    },
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )

        images, video_inputs, video_kwargs = process_vision_info(
            conversation,
            return_video_metadata=True,
            return_video_kwargs=True,
        )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self._processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = self._pooling_last(
                outputs.last_hidden_state, inputs["attention_mask"],
            )
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        result = self._truncate_and_normalize(embeddings[0], self._dimensions)
        elapsed = time.monotonic() - t0

        if verbose:
            print(
                f"    [verbose] dims={len(result)}, "
                f"inference_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return result

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        self._load_model()

        import torch
        import torch.nn.functional as F

        t0 = time.monotonic()

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Retrieve videos relevant to the query."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query_text}],
            },
        ]

        prompt = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = self._pooling_last(
                outputs.last_hidden_state, inputs["attention_mask"],
            )
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        result = self._truncate_and_normalize(embeddings[0], self._dimensions)
        elapsed = time.monotonic() - t0

        if verbose:
            print(
                f"  [verbose] query embedding: dims={len(result)}, "
                f"inference_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return result

    def dimensions(self) -> int:
        return self._dimensions
