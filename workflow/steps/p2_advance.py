# steps/p2_advance.py
#
# Publish the NO set into p3's queue, then close the batch. Last, because
# publication is the only effect another `wf` invocation can observe.

from pathlib import Path

from workflow import batch, config, fs


NAME = "advance"

DESTINATION = ["p3", "queued"]


def inputs(ctx) -> list[Path]:
    return sorted(ctx.batch_dir.glob(f"{ctx.slug}*.p2.no"))


def outputs(ctx) -> list[Path]:
    return [config.path(ctx.root, DESTINATION) / p.name for p in inputs(ctx)]


def is_done(ctx) -> bool:
    return not ctx.batch_dir.exists()


def run_step(ctx) -> None:
    destination = config.path(ctx.root, DESTINATION)
    for produced in inputs(ctx):
        fs.move_into(produced, destination, ctx.force)
    batch.finish_ctx(ctx)
