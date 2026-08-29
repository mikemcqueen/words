# steps/p1_extract.py
#
# Split the bundle's evaluated pairs into YES and NO sets, produced *into the
# bundle directory*. Publication is `advance`'s job, not this step's.

import tempfile

from pathlib import Path

from src.filter import filter_results
from workflow import bundle, config, fs, log, names, setops


NAME = "extract"


def result(ctx) -> Path:
    # By the bundle's own prefix rather than a bare `*.jsonl`: everything in a
    # bundle directory is prefixed by the directory name, so asking for that
    # prefix makes a result belonging to some other bundle a miss instead of
    # the file this step reads. Safe as a glob for the same reason
    # `queue_names` is -- `check_name` has ruled out `*?[]` in a bundle name.
    return bundle.one(ctx.bundle_dir, f"{ctx.bundle_name}*.jsonl")


def produced_bundle_name(ctx) -> str:
    # The bundle's own name plus the band it was filtered at, and nothing from
    # the result file. evalpair appends its own tag/prompt/host suffixes to the
    # name it was handed, so the result stem is a string the workflow does not
    # control: it carried `_` into a p2 queue name that `check_name` then
    # refused, stalling the bundle at `wf eval p2` with nothing to rename. The
    # jsonl never leaves the bundle directory, so its spelling need not be one
    # a name can be made of.
    return names.bundle_name(ctx.bundle_name, ctx.pmin, ctx.prange)


def inputs(ctx) -> list[Path]:
    # The whole archived corpus plus this bundle's own result, which `archive`
    # has not moved into done/out yet -- under the old ordering it already had.
    done_out = config.path(ctx.root, ["p1", "done", "out"])
    return sorted(done_out.glob("*.jsonl")) + [result(ctx)]


def outputs(ctx) -> list[Path]:
    return [ctx.bundle_dir / names.artifact(
                produced_bundle_name(ctx), "p1", "yes"),
            ctx.bundle_dir / names.artifact(
                produced_bundle_name(ctx), "p1", "no")]


def is_done(ctx) -> bool:
    # Not the default -- every output in place -- because a bundle with no
    # result in it has nothing to extract *from*: a directory `archive` has
    # already emptied, or one that never held a result at all. Completing
    # either is a no-op close rather than an error, and only the absence of the
    # jsonl says so. The output names no longer depend on it: they are rendered
    # from the Context, which nothing moves.
    if bundle.at_most_one(ctx.bundle_dir, f"{ctx.bundle_name}*.jsonl") is None:
        return True
    return all(p.exists() for p in outputs(ctx))


def run_step(ctx) -> None:
    corpus = inputs(ctx)
    restrict = bundle.source(ctx)
    yes_path, no_path = outputs(ctx)
    log.info(f"found {fs.line_count(restrict):,} source pairs")

    for yes, dest in ((True, yes_path), (False, no_path)):
        with tempfile.NamedTemporaryFile(mode="w", prefix="wf-p1-extract-",
                                         suffix=".pairs") as matches:
            filter_results(corpus, yes, matches, pairs_path=str(restrict),
                           pmin=ctx.pmin, prng=ctx.prange,
                           report_pair_load=False)
            matches.flush()
            setops.merge([matches.name], dest)

        label = (f"{round(ctx.pmin * 100)}-"
                 f"{round((ctx.pmin + ctx.prange) * 100)}% YES"
                 if yes else "NO")
        log.info(f"filtered {fs.line_count(dest):,} {label} pairs")
