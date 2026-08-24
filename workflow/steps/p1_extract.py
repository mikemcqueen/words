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
    return bundle.one(ctx.bundle_dir, "*.jsonl")


def produced_bundle_name(ctx) -> str:
    # The produced bundle name comes from the p1 result file. The current
    # bundle directory is named for the source pairs file, which is a prefix of
    # it but not the same string.
    return names.bundle_name(result(ctx).stem, ctx.pmin, ctx.prange)


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
    # Output names are rendered from the result stem, so once `archive` has
    # moved the result out there is nothing left to render from -- and nothing
    # left to do either.
    if not any(ctx.bundle_dir.glob("*.jsonl")):
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
