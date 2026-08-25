# steps/merge.py
#
# Fold this bundle's evaluated pairs into the phase's done-set. Shared by both
# phases: only the slot differs, and that comes off the Context.

from pathlib import Path

from workflow import bundle, setops


NAME = "merge"


def inputs(ctx) -> list[Path]:
    return [bundle.evaluated(ctx)]


def outputs(ctx) -> list[Path]:
    return [bundle.done_pairs(ctx)]


def is_done(ctx) -> bool:
    # The done-set is shared across every bundle, so its existence says nothing
    # about *this* bundle, and union under `sort -u` is idempotent -- so this
    # step would happily always run. What it cannot do is run *after*
    # `archive`, which relocates its input into done/. Idempotent is not the
    # same as always runnable, and placement is what answers the difference:
    # if the bundle no longer holds its input, archive ran, so merge did too.
    return not bundle.has_source(ctx)


def run_step(ctx) -> None:
    setops.fold(bundle.evaluated(ctx), bundle.done_pairs(ctx))
