# steps/filter.py
#
# Filter an archived p1 result corpus into a single YES pair set.

import tempfile

from pathlib import Path

from src.filter import filter_results
from workflow import fs, log, select, setops


NAME = "filter"

SLOT = ["p1", "done", "out"]


def inputs(ctx) -> list[Path]:
    if ctx.results_dir is not None:
        fs.raise_if_not_dir(ctx.results_dir)
        paths = sorted(p for p in ctx.results_dir.glob("*.jsonl") if p.is_file())
        if not paths:
            raise ValueError(f"no *.jsonl files in {ctx.results_dir}")
        return paths
    # The selector arrives on the Context, so this resolves from ctx alone --
    # no previous step has to hand the path list forward.
    return select.select(ctx.root, SLOT, ctx.selector, glob="*.jsonl")


def outputs(ctx) -> list[Path]:
    return [ctx.dest]


def is_done(ctx) -> bool:
    # ctx.dest is a name the *user* chose with -o, not a rendered artifact: its
    # existence says nothing about which band produced it, so the default
    # "outputs exist => skip" would silently keep a stale file and report
    # success. Never skip; run_step refuses to clobber unless -f.
    return False


def run_step(ctx) -> None:
    paths = inputs(ctx)
    fs.raise_if_not_dir(ctx.dest.parent)
    if not ctx.force:
        fs.raise_if_exists(ctx.dest)

    # Emit a set by construction: filter into scratch, then place atomically.
    with tempfile.NamedTemporaryFile(mode="w", prefix="wf-filter-",
                                     suffix=".pairs") as matches:
        filter_results(paths, True, matches,
                       pairs_path=(str(ctx.pairs_path)
                                   if ctx.pairs_path is not None else None),
                       pmin=ctx.pmin, prng=ctx.prange, use_max=False)
        matches.flush()
        setops.merge([matches.name], ctx.dest)

    log.info(f"filtered {len(paths)} p1 result file(s) → {ctx.dest.name}")
