# setops.py
#
# Set primitives over sorted-unique line files.
#
# A sorted-unique line file *is* a set here, and the collation is load-bearing:
# `comm -23` assumes both inputs are ordered the way it orders them. Every
# shell-out therefore runs under LC_ALL=C, so the ordering is a property of the
# operation rather than of whatever locale the caller happened to run under. A
# set merged under one locale and diffed under another yields a silently wrong
# difference, not an error.
#
# Every operation is a write aside followed by a rename, and the two halves are
# separable. An immediate operation does both and is what most callers want. A
# caller placing several results at once -- the dictionary rebuild, which
# publishes two archives and two derived sets -- stages them all, checks every
# destination, and only then renames, so a validation or set operation failure
# cannot leave half a batch published.

import filecmp
import os
import subprocess

from dataclasses import dataclass
from pathlib import Path


def _c_env() -> dict:
    return {**os.environ, "LC_ALL": "C"}


@dataclass(frozen=True)
class Placement:
    """Where an operation put its result, and whether it wrote there.

    `replaced` is false only where stable_mtime found the result byte-identical
    to what was already at `path` and left it alone -- which is the one thing a
    caller reporting on a rebuild cannot work out from the result: diff and
    common are not monotone, so an unchanged line count proves nothing.
    """

    path: Path
    replaced: bool


@dataclass(frozen=True)
class Staged:
    """A result written aside, waiting for the rename that publishes it.

    `replaced` records what the comparison already decided during staging, so
    `place` is a rename and nothing else and a caller can drop the no-ops from
    a batch without reading either file a second time.
    """

    path: Path
    dst: Path
    replaced: bool


def _tmp(dst: Path) -> Path:
    return dst.with_name(dst.name + ".tmp")


def _unchanged(staged: Path, dst: Path) -> bool:
    """Is what was staged byte-identical to what is already at dst?

    filecmp caches by (path, size, mtime), and an immediate placement reuses
    one temp path per destination -- so two placements to one dst in a run can
    collide on a coarse mtime and reuse the earlier answer without reading
    either file. Under diff or common, where equal size proves nothing, that
    would silently skip a needed write.
    """
    if not dst.exists():
        return False
    filecmp.clear_cache()
    return filecmp.cmp(staged, dst, shallow=False)


def stage(argv: list[str], dst: Path, staged: Path,
          stable_mtime: bool = False) -> Staged:
    """Run argv with stdout captured into staged, for a later place().

    Everything that can fail -- the subprocess, the write, the content compare
    -- happens here; place() is the rename alone. The staged path is the
    caller's to choose, so a batch can stage into one private directory on the
    destination filesystem and leave the destinations untouched until it is
    whole. Disposal is the caller's too: a failure here leaves the staged file
    behind, for the staging directory to discard wholesale or the immediate
    placement to unlink.
    """
    with staged.open("w") as f:
        subprocess.run(argv, stdout=f, env=_c_env(), check=True)
    replaced = not (stable_mtime and _unchanged(staged, dst))
    return Staged(path=staged, dst=dst, replaced=replaced)


def stage_text(text: str, dst: Path, staged: Path,
               stable_mtime: bool = False) -> Staged:
    """stage, for a result that is written rather than run."""
    staged.write_text(text)
    replaced = not (stable_mtime and _unchanged(staged, dst))
    return Staged(path=staged, dst=dst, replaced=replaced)


def place(staged: Staged) -> Placement:
    """Publish one staged result. An atomic rename, or nothing at all."""
    if staged.replaced:
        staged.path.replace(staged.dst)
    return Placement(path=staged.dst, replaced=staged.replaced)


def _place_report(argv: list[str], dst: Path,
                  stable_mtime: bool = False) -> Placement:
    """Run argv and place the result at dst atomically, saying what it did.

    Writing to a sibling temp and renaming means dst is never observed
    half-written, and a failure leaves the previous dst intact. It also lets dst
    itself appear in argv, which is what makes merging into an existing set safe
    without the rename-aside-and-restore dance it replaces.

    stable_mtime keeps dst untouched when the result is byte-identical to what
    is already there, for a destination whose mtime is read as "the set changed"
    rather than "something wrote here". It is off by default because the compare
    costs a read of both files, which is not worth paying on an accumulator
    nobody dates -- see fold. The comparison has to be over content: this is the
    generic placement step, and diff and common are not monotone, so equal line
    counts prove nothing about them.
    """
    tmp = _tmp(dst)
    try:
        result = place(stage(argv, dst, tmp, stable_mtime=stable_mtime))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.unlink(missing_ok=True)
    return result


def _place(argv: list[str], dst: Path, stable_mtime: bool = False) -> Path:
    return _place_report(argv, dst, stable_mtime=stable_mtime).path


def _merge_argv(srcs) -> list[str]:
    srcs = [Path(s) for s in srcs]
    if not srcs:
        raise ValueError("merge requires at least one source")
    return ["sort", "-u", *(str(s) for s in srcs)]


def _diff_argv(a: Path, b: Path) -> list[str]:
    return ["comm", "-23", str(a), str(b)]


def merge(srcs, dst: Path, stable_mtime: bool = False) -> Path:
    """Union: the sorted-unique set of every line in srcs, written to dst.

    One source is the degenerate case -- `sort -u` on a single file -- which is
    why there is no separate normalize operation.
    """
    return merge_report(srcs, dst, stable_mtime=stable_mtime).path


def merge_report(srcs, dst: Path, stable_mtime: bool = False) -> Placement:
    """merge, for the caller that has to say whether the set moved."""
    return _place_report(_merge_argv(srcs), Path(dst), stable_mtime=stable_mtime)


def stage_merge(srcs, dst: Path, staged: Path,
                stable_mtime: bool = False) -> Staged:
    """merge into a staged file, for a batch that places several at once."""
    return stage(_merge_argv(srcs), Path(dst), staged,
                 stable_mtime=stable_mtime)


def fold(src: Path, dst: Path, stable_mtime: bool = False) -> Path:
    """Union src into dst, whether or not dst already exists.

    The accumulator case: every standing set in the workflow -- a phase's
    done-set, a classified set -- is grown by folding one file into it. `merge`
    can take dst as one of its own sources because `_place` writes aside and
    renames, but only once dst is there to be read; this is that guard, written
    once instead of at each accumulator.
    """
    src, dst = Path(src), Path(dst)
    return merge([dst, src] if dst.exists() else [src], dst,
                 stable_mtime=stable_mtime)


def diff(a: Path, b: Path, dst: Path, stable_mtime: bool = False) -> Path:
    """Difference: the lines of a that are not in b. Both inputs must be sets."""
    return diff_report(a, b, dst, stable_mtime=stable_mtime).path


def diff_report(a: Path, b: Path, dst: Path,
                stable_mtime: bool = False) -> Placement:
    """diff, for the caller that has to say whether the set moved."""
    return _place_report(_diff_argv(a, b), Path(dst), stable_mtime=stable_mtime)


def stage_diff(a: Path, b: Path, dst: Path, staged: Path,
               stable_mtime: bool = False) -> Staged:
    """diff into a staged file, for a batch that places several at once."""
    return stage(_diff_argv(a, b), Path(dst), staged, stable_mtime=stable_mtime)


def common(a: Path, b: Path, dst: Path, stable_mtime: bool = False) -> Path:
    """Intersection: the lines in both a and b. Both inputs must be sets."""
    return _place(["comm", "-12", str(a), str(b)], Path(dst),
                  stable_mtime=stable_mtime)
