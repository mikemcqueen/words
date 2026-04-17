"""Optional native fast path for aligned eval-result chunk loading."""

from __future__ import annotations

_IMPORT_ERROR = None

try:
    from _compare_native import (
        AlignedJsonlReader,
        LABEL_NO,
        LABEL_UNKNOWN,
        LABEL_YES,
        ProjectedAlignedJsonlReader,
    )
except ImportError as exc:  # pragma: no cover - exercised when extension is absent
    AlignedJsonlReader = None
    ProjectedAlignedJsonlReader = None
    LABEL_UNKNOWN = 0
    LABEL_NO = 1
    LABEL_YES = 2
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


def iter_projected_blocks(files_list, chunk_size=100):
    """Return a native iterator over compact projected blocks."""
    require_native()
    return ProjectedAlignedJsonlReader(files_list, chunk_size)
