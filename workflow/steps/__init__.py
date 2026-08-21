# steps
#
# The operation protocol and its runner.
#
# A *step* takes only a Context and derives every path it touches from it.
# Steps are what a STEPS list holds. A *primitive* (select, merge, diff,
# filter_results) takes arguments and returns or writes what it is told; steps
# call primitives, and primitives never appear in a STEPS list.
#
# A step never receives a value from the step before it. Per-step is_done
# skipping makes any in-memory channel unsound: skip the producer and the
# value it would have passed forward is simply absent, on exactly the resume
# path the protocol exists to support. Steps communicate through the
# filesystem at rendered paths, or not at all -- which is what inputs(ctx) and
# outputs(ctx) make explicit.
#
#     NAME: str
#     def inputs(ctx)  -> list[Path]
#     def outputs(ctx) -> list[Path]
#     def run_step(ctx) -> None
#     def is_done(ctx) -> bool        # optional; defaults to "outputs exist"

from workflow import log


def is_done(step, ctx) -> bool:
    if hasattr(step, "is_done"):
        return step.is_done(ctx)
    outputs = step.outputs(ctx)
    return bool(outputs) and all(p.exists() for p in outputs)


def run_steps(steps, ctx) -> int:
    """Run a recipe, resuming past whatever is already in place.

    Re-running after a mid-way failure continues where it stopped;
    -f/--force means "ignore is_done and overwrite".
    """
    for step in steps:
        if not ctx.force and is_done(step, ctx):
            log.info(f"skip {step.NAME}: already done")
            continue
        step.run_step(ctx)
    return 0
