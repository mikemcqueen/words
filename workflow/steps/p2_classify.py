# steps/p2_classify.py
#
# Fold this bundle's manually-confirmed YES pairs into the global classified set.
# Carried over as-is from _process_yes_pairs.

from pathlib import Path

from workflow import config


NAME = "classify"


def classified_yes(ctx) -> Path:
    return config.classified(ctx.root, "yes")


def inputs(ctx) -> list[Path]:
    return [ctx.artifact("p2", "yes")]


def outputs(ctx) -> list[Path]:
    return [classified_yes(ctx)]


def is_done(ctx) -> bool:
    # Global across every bundle, like the phase done-set: its existence says
    # nothing about this bundle, and union is idempotent -- but see merge, whose
    # answer this mirrors. `archive` relocates this fold's input into
    # p2/done/out, so the input's absence is the record that the fold ran.
    # Tested directly rather than via bundle.has_source: extract_yes writes this
    # artifact unconditionally, so at this point in the recipe it is present
    # unless archive has taken it.
    return not ctx.artifact("p2", "yes").exists()


def run_step(ctx) -> None:
    config.fold_classified(ctx.root, "yes", ctx.artifact("p2", "yes"))
