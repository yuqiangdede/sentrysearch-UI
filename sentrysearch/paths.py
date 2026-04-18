"""Central project-local path definitions."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / ".sentrysearch"
DB_DIR = RUNTIME_DIR / "db"
ENV_PATH = RUNTIME_DIR / ".env"
TEMP_DIR = RUNTIME_DIR / "tmp"
HF_CACHE_DIR = RUNTIME_DIR / "hf-cache"
MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads" / "videos"
CLIPS_DIR = PROJECT_ROOT / "clips_output"


def ensure_dir(path: Path) -> Path:
    """Create *path* and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(path: str | Path) -> Path:
    """Resolve *path* and require it to stay under the project root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if candidate != PROJECT_ROOT and PROJECT_ROOT not in candidate.parents:
        raise ValueError(f"Path must stay inside the project root: {path}")
    return candidate
