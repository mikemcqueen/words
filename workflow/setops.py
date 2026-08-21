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

import os
import subprocess

from pathlib import Path


def _c_env() -> dict:
    return {**os.environ, "LC_ALL": "C"}


def _place(argv: list[str], dst: Path) -> Path:
    """Run argv with stdout captured, then place the result at dst atomically.

    Writing to a sibling temp and renaming means dst is never observed
    half-written, and a failure leaves the previous dst intact. It also lets dst
    itself appear in argv, which is what makes merging into an existing set safe
    without the rename-aside-and-restore dance it replaces.
    """
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        with tmp.open("w") as f:
            subprocess.run(argv, stdout=f, env=_c_env(), check=True)
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dst


def merge(srcs, dst: Path) -> Path:
    """Union: the sorted-unique set of every line in srcs, written to dst.

    One source is the degenerate case -- `sort -u` on a single file -- which is
    why there is no separate normalize operation.
    """
    srcs = [Path(s) for s in srcs]
    if not srcs:
        raise ValueError("merge requires at least one source")
    return _place(["sort", "-u", *(str(s) for s in srcs)], Path(dst))


def diff(a: Path, b: Path, dst: Path) -> Path:
    """Difference: the lines of a that are not in b. Both inputs must be sets."""
    return _place(["comm", "-23", str(a), str(b)], Path(dst))
