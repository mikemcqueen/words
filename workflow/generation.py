# generation.py
#
# The generation clock: the marker beside a derived artifact that says when it
# was last made, as opposed to when its contents last changed.
#
# Two clocks, because one file cannot answer both questions. An artifact placed
# with stable_mtime keeps its content mtime through a byte-identical
# regeneration -- which is what stops a no-op from cascading into hours of
# downstream work -- and that same mtime therefore cannot say whether the
# artifact has been generated since its input moved. The marker answers the
# second question and advances on every gen, no-op or not.
#
# Shared rather than BEST's own: top.segments and the derived dictionary are
# the same kind of object and need the same pair of clocks, and a second
# implementation of the marker name would be a second convention.

from pathlib import Path


def stamp(path: Path) -> Path:
    """The generation marker beside an artifact, written by every gen."""
    return path.with_name(f".{path.name}.gen")


def generated(path: Path) -> Path:
    """The path whose mtime dates an artifact against its own inputs.

    An artifact placed with stable_mtime keeps its mtime through a
    byte-identical regeneration, which is what stops a no-op from cascading
    into an hours-long DFS downstream. That same mtime cannot also answer
    "has this been generated since its input moved?", because the answer it
    gives never changes: an input that moves and yields identical content
    reports stale forever, and the gen offered to clear it is the one write
    stable_mtime suppresses. The marker answers that question -- it advances
    on every gen, no-op or not -- and the artifact's own mtime goes on
    answering the first for whatever reads it downstream.

    Absent the marker the artifact dates itself, which is what a tree built
    before the marker existed, or an artifact placed by hand, will do.
    """
    marker = stamp(path)
    return marker if marker.exists() else path


def mark_generated(path: Path, text: str = "") -> None:
    """Record a successful generation, optionally with what it was made from.

    write_text rather than touch: a marker that also carries contents has to
    advance its mtime whether or not those contents changed, or the generation
    clock would stall exactly where stable_mtime stalls the content clock.
    """
    stamp(path).write_text(text)
