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

import filecmp
import os
import subprocess

from pathlib import Path


def _c_env() -> dict:
    return {**os.environ, "LC_ALL": "C"}


def _place(argv: list[str], dst: Path, stable_mtime: bool = False) -> Path:
    """Run argv with stdout captured, then place the result at dst atomically.

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
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        with tmp.open("w") as f:
            subprocess.run(argv, stdout=f, env=_c_env(), check=True)
        if stable_mtime and dst.exists():
            # filecmp caches by (path, size, mtime), and every placement here
            # uses the same tmp path -- so two placements to one dst in a run
            # can collide on a coarse mtime and reuse the earlier answer
            # without reading either file. Under diff or common, where equal
            # size proves nothing, that would silently skip a needed write.
            filecmp.clear_cache()
            if filecmp.cmp(tmp, dst, shallow=False):
                tmp.unlink()
                return dst
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dst


def merge(srcs, dst: Path, stable_mtime: bool = False) -> Path:
    """Union: the sorted-unique set of every line in srcs, written to dst.

    One source is the degenerate case -- `sort -u` on a single file -- which is
    why there is no separate normalize operation.
    """
    srcs = [Path(s) for s in srcs]
    if not srcs:
        raise ValueError("merge requires at least one source")
    return _place(["sort", "-u", *(str(s) for s in srcs)], Path(dst),
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
    return _place(["comm", "-23", str(a), str(b)], Path(dst),
                  stable_mtime=stable_mtime)


def common(a: Path, b: Path, dst: Path, stable_mtime: bool = False) -> Path:
    """Intersection: the lines in both a and b. Both inputs must be sets."""
    return _place(["comm", "-12", str(a), str(b)], Path(dst),
                  stable_mtime=stable_mtime)
