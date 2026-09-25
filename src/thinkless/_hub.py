"""Model downloads from the Hugging Face Hub, made reliable on Windows."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from .logs import get_logger

__all__ = ["DEFAULT_CHECKPOINTS", "cached_checkpoints", "ensure_downloaded"]

logger = get_logger("hub")

# What `thinkless demo` and the default providers download on first use.
DEFAULT_CHECKPOINTS = {
    "fastino/gliner2.5-base-v1": "GLiNER provider",
    "convaiinnovations/laya": "Laya provider",
    "Qwen/Qwen3-1.7B": "local LLM (--llm local)",
}


# Weight formats ThinkLess never loads. Skipping them keeps the serial download
# as small as the one the model library would have made.
IGNORED = [
    "*.onnx",
    "onnx/*",
    "*.msgpack",
    "*.h5",
    "*.ot",
    "*.tflite",
    "*.gguf",
    "flax_model*",
    "tf_model*",
    "rust_model*",
    "coreml/*",
    "openvino/*",
]


def _is_local_path(repo_id: str) -> bool:
    return Path(repo_id).exists()


def ensure_downloaded(repo_id: str) -> None:
    """Download ``repo_id`` into the Hugging Face cache before a library loads it.

    On Windows the Hub client downloads files in parallel and can hit a race
    when it creates cache symlinks (``WinError 1314``). Downloading serially
    first avoids it; the model library then finds everything in the cache.
    Elsewhere this is a no-op and the library downloads as usual. Local paths
    and offline mode (``HF_HUB_OFFLINE=1``) are left alone.
    """
    if platform.system() != "Windows" or _is_local_path(repo_id):
        return
    if os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "True"):
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - installed with every local extra
        return
    try:
        snapshot_download(repo_id, max_workers=1, ignore_patterns=IGNORED)
    except Exception as exc:  # the library's own loader reports the real error
        logger.warning(
            "pre-download of %s failed (%s); falling back to the library loader", repo_id, exc
        )


def cached_checkpoints() -> dict[str, float | None]:
    """Size in GB of each default checkpoint in the local cache, or ``None`` if absent."""
    sizes: dict[str, float | None] = dict.fromkeys(DEFAULT_CHECKPOINTS)
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return sizes
    try:
        info = scan_cache_dir()
    except Exception:
        return sizes
    for repo in info.repos:
        if repo.repo_id in sizes and repo.repo_type == "model":
            sizes[repo.repo_id] = round(repo.size_on_disk / 1e9, 2)
    return sizes
