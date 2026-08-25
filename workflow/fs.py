# fs.py

import errno
import fnmatch
from pathlib import Path

def not_a_directory_error(path: Path): 
    return NotADirectoryError(errno.ENOTDIR, "not a directory", path)


def not_a_file_error(path: Path): 
    return ValueError(f"not a regular file: {path}")


def file_not_found_error(path: Path): 
    return FileNotFoundError(errno.ENOENT, "file not found", path)


def file_already_exists_error(path: Path): 
    return ValueError(f"file already exists: {path}")


def raise_if_not_exists(path: Path):
    if not path.exists():
        raise file_not_found_error(path)


def raise_if_exists(path: Path):
    if path.exists():
        raise file_already_exists_error(path)


def raise_if_not_dir(path: Path):
    raise_if_not_exists(path)
    if not path.is_dir():
        raise not_a_directory_error(path)


def raise_if_not_file(path: Path):
    raise_if_not_exists(path)
    if not path.is_file():
        raise not_a_file_error(path)
    

def raise_if_any_exist(paths: list[Path]):
    for path in paths:
        raise_if_exists(path)


def raise_if_any_not_file(paths: list[Path]):
    for path in paths:
        raise_if_not_file(path)


def spell(patterns) -> str:
    """How to name one pattern or several in a diagnostic."""
    return patterns if isinstance(patterns, str) else " or ".join(patterns)


def matches(name: str, patterns) -> bool:
    """Does a filename match any of patterns? The name-level twin of globs.

    Same one-or-several convention, for the caller that already holds the file
    and only needs to know whether it is the shape that was asked for.
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def globs(directory: Path, patterns) -> list[Path]:
    """The files in directory matching any of patterns, sorted and deduplicated.

    Takes one pattern or several, because a slot can hold more than one
    canonical name shape and no caller should have to care which one arrived.
    Deduplicated because overlapping patterns are legal and a file found twice
    is still one file.
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    found = {p for pattern in patterns
             for p in directory.glob(pattern) if p.is_file()}
    return sorted(found)


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    return n_lines


def move_into(src: Path, dst_dir: Path, force: bool = False) -> Path:
    """Rename src into dst_dir, keeping its name. Atomic on one filesystem."""
    dst = dst_dir / src.name
    if not force:
        raise_if_exists(dst)
    src.rename(dst)
    return dst


# A step that moves several things is restartable only if each move can tell
# "a previous run already did this" from "this is missing". Gone from the
# source *and* absent at the destination is the second, and still raises.

def move_into_once(src: Path, dst_dir: Path, force: bool = False) -> Path:
    """move_into, tolerant of a previous run having already done it."""
    if not src.exists():
        raise_if_not_exists(dst_dir / src.name)
        return dst_dir / src.name
    return move_into(src, dst_dir, force)


def rename_once(src: Path, dst: Path, force: bool = False) -> Path:
    """Rename src to dst, tolerant of a previous run having already done it."""
    if not src.exists():
        raise_if_not_exists(dst)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force:
        raise_if_exists(dst)
    src.rename(dst)
    return dst
