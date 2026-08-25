# steps/p1_archive.py
#
# Move the bundle's *inputs* into done/{in,out}. The artifacts this bundle
# produced stay put -- publishing those is `advance`'s job, and it runs last.

from pathlib import Path

from workflow import config, fs, names


NAME = "archive"


def _done(ctx, inout: str) -> Path:
    return config.path(ctx.root, ["p1", "done", inout])


# What moves where. Globbed rather than rendered: the result file's name is
# evalpair's, not ours, and the bundle directory is already the namespace.
MOVES = (("*.jsonl", "out"), (names.queue_globs("p1"), "in"))


def inputs(ctx) -> list[Path]:
    return [p for glob, _ in MOVES for p in fs.globs(ctx.bundle_dir, glob)]


def outputs(ctx) -> list[Path]:
    return [_done(ctx, inout) / p.name
            for glob, inout in MOVES for p in fs.globs(ctx.bundle_dir, glob)]


def is_done(ctx) -> bool:
    # This step's whole job is to empty the bundle of its inputs, so their
    # absence is the answer -- nothing has to be rendered, and it stays False
    # until *every* move has landed rather than just the first.
    return not inputs(ctx)


def run_step(ctx) -> None:
    # Each move is independent: take whatever is still here rather than demand
    # it, so a retry after a partial move finishes the rest instead of crashing
    # on what already left. `extract` has already enforced the one-result rule.
    for glob, inout in MOVES:
        for found in fs.globs(ctx.bundle_dir, glob):
            fs.move_into(found, _done(ctx, inout), ctx.force)

    # The .filtered derivative is scratch: the original is what gets archived.
    for scratch in ctx.bundle_dir.glob("*.filtered"):
        scratch.unlink()
