"""Download project-local embedding models into ./models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from sentrysearch.paths import resolve_project_path

os.environ.setdefault("HF_HOME", str(resolve_project_path(".sentrysearch/hf-cache")))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(resolve_project_path(".sentrysearch/hf-cache/hub")))
os.environ.setdefault("TRANSFORMERS_CACHE", str(resolve_project_path(".sentrysearch/hf-cache/transformers")))

MODELS: dict[str, tuple[str, str]] = {
    "qwen2b": ("Qwen/Qwen3-VL-Embedding-2B", "Qwen3-VL-Embedding-2B"),
    "qwen8b": ("Qwen/Qwen3-VL-Embedding-8B", "Qwen3-VL-Embedding-8B"),
    "qwen3vl-reranker-2b": ("Qwen/Qwen3-VL-Reranker-2B", "Qwen3-VL-Reranker-2B"),
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SentrySearch models into the project-local models directory.",
    )
    parser.add_argument(
        "--model",
        choices=["qwen2b", "qwen8b", "qwen3vl-reranker-2b", "all"],
        default="qwen2b",
        help="Which model to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Destination directory inside the project.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, or commit.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional Hugging Face endpoint or mirror URL, e.g. https://hf-mirror.com.",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Model hub to download from.",
    )
    return parser.parse_args(argv)


def _import_huggingface_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - simple dependency hint
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install the project dependencies "
            "in .venv first."
        ) from exc
    return snapshot_download


def _import_modelscope_snapshot_download():
    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover - simple dependency hint
        raise SystemExit(
            "Missing dependency: modelscope. Install the project dependencies "
            "in .venv first."
        ) from exc
    return snapshot_download


def _resolve_endpoint(cli_endpoint: str | None) -> str | None:
    """Resolve the download endpoint from CLI or environment."""
    if cli_endpoint:
        return cli_endpoint
    return os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_ENDPOINT")


def _download_one(
    model_key: str,
    output_dir: Path,
    revision: str | None,
    endpoint: str | None,
    source: str,
) -> Path:
    repo_id, folder_name = MODELS[model_key]
    target_dir = output_dir / folder_name
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if source == "modelscope":
        print(f"Downloading {repo_id} via ModelScope -> {target_dir}")
        snapshot_download = _import_modelscope_snapshot_download()
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            revision=revision,
        )
    else:
        snapshot_download = _import_huggingface_snapshot_download()
        if endpoint:
            print(f"Downloading {repo_id} via {endpoint} -> {target_dir}")
        else:
            print(f"Downloading {repo_id} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            revision=revision,
            endpoint=endpoint,
        )
    return target_dir


def _validate_source(source: str, endpoint: str | None) -> None:
    if source == "modelscope" and endpoint:
        print("Warning: --endpoint is ignored when --source modelscope.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    output_dir = resolve_project_path(args.output_dir)
    endpoint = _resolve_endpoint(args.endpoint)
    _validate_source(args.source, endpoint)

    keys = list(MODELS) if args.model == "all" else [args.model]
    for key in keys:
        _download_one(key, output_dir, args.revision, endpoint, args.source)

    print(f"Done. Models are stored under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
