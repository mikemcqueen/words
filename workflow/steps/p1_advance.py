# steps/p1_advance.py
#
# Publish what the batch produced into the next phases' queues, then close the
# batch. Last, because publication is the only effect another `wf` invocation
# can observe -- nothing downstream should see this batch until everything
# else about it has landed.

from pathlib import Path

from workflow import batch, config, fs


NAME = "advance"

# produced kind -> the slot that consumes it
DESTINATIONS = {"yes": ["p2", "queued"], "no": ["p3", "queued"]}


# What `extract` produced is matched by kind, not by the batch's name: extract
# renders its output names from the evalpair result's stem, which nothing
# forces to match the directory. Everything in the directory belongs to this
# batch anyway, so the directory is the only scoping needed.
def inputs(ctx) -> list[Path]:
    found = []
    for kind in DESTINATIONS:
        found += sorted(ctx.batch_dir.glob(f"*.p1.{kind}"))
    return found


def outputs(ctx) -> list[Path]:
    return [config.path(ctx.root, slot) / p.name
            for kind, slot in DESTINATIONS.items()
            for p in sorted(ctx.batch_dir.glob(f"*.p1.{kind}"))]


def is_done(ctx) -> bool:
    # The batch directory exists iff work is in flight, so its absence is the
    # whole answer -- and it is one stat.
    return not ctx.batch_dir.exists()


def run_step(ctx) -> None:
    for kind, slot in DESTINATIONS.items():
        destination = config.path(ctx.root, slot)
        for produced in sorted(ctx.batch_dir.glob(f"*.p1.{kind}")):
            fs.move_into(produced, destination, ctx.force)

    batch.finish(ctx)
