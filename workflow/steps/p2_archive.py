# steps/p2_archive.py
#
# Move the batch's input and its manual-review output into done/. The NO set
# stays put -- publishing it is `advance`'s job.

from pathlib import Path

from workflow import batch, config, fs
from workflow.steps import p2_retrieve


NAME = "archive"


def _done(ctx, *parts: str) -> Path:
    return config.path(ctx.root, ["p2", "done", *parts])


def enex_archive(ctx) -> Path:
    # Per-batch rather than flat: enex is never queried across batches, so the
    # "flat where you query" rule does not apply to it.
    return _done(ctx, "out", "enex") / ctx.slug


def inputs(ctx) -> list[Path]:
    return (sorted(ctx.batch_dir.glob(batch.INPUT_GLOB["p2"]))
            + [ctx.artifact("p2", "yes"), p2_retrieve.enex_dir(ctx)])


def outputs(ctx) -> list[Path]:
    return [_done(ctx, "in") / p.name
            for p in sorted(ctx.batch_dir.glob(batch.INPUT_GLOB["p2"]))] + [
        _done(ctx, "out") / ctx.artifact("p2", "yes").name,
        enex_archive(ctx)]


def is_done(ctx) -> bool:
    # As in p1_archive: emptying the batch is the job, so the absence of what
    # it moves is the answer, and it stays False until every move has landed.
    return not (batch.has_source(ctx)
                or ctx.artifact("p2", "yes").exists()
                or p2_retrieve.enex_dir(ctx).exists())


def run_step(ctx) -> None:
    # Each move is independent, so a retry after a partial move finishes the
    # rest instead of crashing on whichever one already left.
    fs.move_into_once(ctx.artifact("p2", "yes"), _done(ctx, "out"), ctx.force)
    fs.rename_once(p2_retrieve.enex_dir(ctx), enex_archive(ctx), ctx.force)

    for found in sorted(ctx.batch_dir.glob(batch.INPUT_GLOB["p2"])):
        fs.move_into(found, _done(ctx, "in"), ctx.force)

    # The .filtered derivative is scratch: the original is what gets archived.
    for scratch in ctx.batch_dir.glob("*.filtered"):
        scratch.unlink()
