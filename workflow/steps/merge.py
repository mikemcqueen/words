# steps/merge.py
#
# Fold this batch's evaluated pairs into the phase's done-set. Shared by both
# phases: only the slot differs, and that comes off the Context.

from pathlib import Path

from workflow import batch, setops


NAME = "merge"


def inputs(ctx) -> list[Path]:
    return [batch.evaluated(ctx)]


def outputs(ctx) -> list[Path]:
    return [batch.done_pairs(ctx)]


def is_done(ctx) -> bool:
    # The done-set is shared across every batch, so its existence says nothing
    # about *this* batch, and union under `sort -u` is idempotent -- so this
    # step would happily always run. What it cannot do is run *after*
    # `archive`, which relocates its input into done/. Idempotent is not the
    # same as always runnable, and placement is what answers the difference:
    # if the batch no longer holds its input, archive ran, so merge did too.
    return not batch.has_source(ctx)


def run_step(ctx) -> None:
    dst = batch.done_pairs(ctx)
    src = batch.evaluated(ctx)
    setops.merge([dst, src] if dst.exists() else [src], dst)
