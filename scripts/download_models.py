"""Download project-local embedding models into ./models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - simple dependency hint
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install the project dependencies "
        "in .venv first."
    ) from exc


MODELS: dict[str, tuple[str, str]] = {
    "qwen2b": ("Qwen/Qwen3-VL-Embedding-2B", "Qwen3-VL-Embedding-2B"),
    "qwen8b": ("Qwen/Qwen3-VL-Embedding-8B", "Qwen3-VL-Embedding-8B"),
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SentrySearch models into the project-local models directory.",
    )
    parser.add_argument(
        "--model",
        choices=["qwen2b", "qwen8b", "all"],
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
    return parser.parse_args(argv)


def _download_one(model_key: str, output_dir: Path, revision: str | None) -> Path:
    repo_id, folder_name = MODELS[model_key]
    target_dir = output_dir / folder_name
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        revision=revision,
    )
    return target_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    output_dir = Path(args.output_dir).resolve()

    keys = list(MODELS) if args.model == "all" else [args.model]
    for key in keys:
        _download_one(key, output_dir, args.revision)

    print(f"Done. Models are stored under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
