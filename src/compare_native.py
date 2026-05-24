"""Native fast path for projected eval-result loading."""

from __future__ import annotations

_IMPORT_ERROR = None

try:
    from ._compare_native import LABEL_NO, LABEL_UNKNOWN, LABEL_YES, ProjectedAlignedJsonlReader
except ImportError as exc:  # pragma: no cover - exercised when extension is absent
    ProjectedAlignedJsonlReader = None
    LABEL_UNKNOWN = 0
    LABEL_NO = 1
    LABEL_YES = 2
    _IMPORT_ERROR = exc


def _native_available() -> bool:
    return ProjectedAlignedJsonlReader is not None


def _require_native() -> None:
    if _native_available():
        return
    detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
    raise RuntimeError(f"native compare loader unavailable{detail}")


def iter_projected_blocks(files_list, chunk_size=100):
    """Return a native iterator over compact projected blocks."""
    _require_native()
    return ProjectedAlignedJsonlReader(files_list, chunk_size)
