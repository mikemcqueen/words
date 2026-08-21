# steps/p1_extract.py
#
# Split the batch's evaluated pairs into YES and NO sets, produced *into the
# batch directory*. Publication is `advance`'s job, not this step's.

import tempfile

from pathlib import Path

from src.filter import filter_results
from workflow import batch, config, names, setops


NAME = "extract"


def result(ctx) -> Path:
    return batch.one(ctx.batch_dir, "*.jsonl")


def produced_slug(ctx) -> str:
    # The batch is the originating p1 result file; the p1 batch *directory* is
    # named for the pairs file, which is a prefix of it but not the same string.
    return names.slug(result(ctx).stem, ctx.pmin, ctx.prange)


def inputs(ctx) -> list[Path]:
    # The whole archived corpus plus this batch's own result, which `archive`
    # has not moved into done/out yet -- under the old ordering it already had.
    done_out = config.path(ctx.root, ["p1", "done", "out"])
    return sorted(done_out.glob("*.jsonl")) + [result(ctx)]


def outputs(ctx) -> list[Path]:
    slug = produced_slug(ctx)
    return [ctx.batch_dir / names.artifact(slug, "p1", "yes"),
            ctx.batch_dir / names.artifact(slug, "p1", "no")]


def is_done(ctx) -> bool:
    # Output names are rendered from the result stem, so once `archive` has
    # moved the result out there is nothing left to render from -- and nothing
    # left to do either.
    if not any(ctx.batch_dir.glob("*.jsonl")):
        return True
    return all(p.exists() for p in outputs(ctx))


def run_step(ctx) -> None:
    corpus = inputs(ctx)
    restrict = batch.source(ctx)
    yes_path, no_path = outputs(ctx)

    for yes, dest in ((True, yes_path), (False, no_path)):
        with tempfile.NamedTemporaryFile(mode="w", prefix="wf-p1-extract-",
                                         suffix=".pairs") as matches:
            filter_results(corpus, yes, matches, pairs_path=str(restrict),
                           pmin=ctx.pmin, prng=ctx.prange)
            matches.flush()
            setops.merge([matches.name], dest)
