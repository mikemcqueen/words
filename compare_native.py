"""Optional native fast path for aligned eval-result chunk loading."""

from __future__ import annotations

_IMPORT_ERROR = None

try:
    from _compare_native import AlignedJsonlReader
except ImportError as exc:  # pragma: no cover - exercised when extension is absent
    AlignedJsonlReader = None
    _IMPORT_ERROR = exc


def native_available() -> bool:
    return AlignedJsonlReader is not None


def require_native() -> None:
    if native_available():
        return
    detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
    raise RuntimeError(f"native compare loader unavailable{detail}")


def iter_aligned_chunks(files_list, chunk_size=100):
    """Return a native iterator over aligned parsed chunks."""
    require_native()
    return AlignedJsonlReader(files_list, chunk_size)
