# steps/p2_classify.py
#
# Fold this bundle's manually-confirmed verdicts into the global classified
# sets.
#
# Both kinds, because both are confirmed. `extract` parses the notes with
# --two-checkboxes, so a NO row is one a reviewer ticked N on -- not one that
# merely failed to be a YES. That makes NO a standing verdict of the same
# standing as YES, and standing verdicts live in classified/. p2 therefore
# publishes nothing onward: what used to go to p3's queue for another look has
# already had the look that settles it.

from pathlib import Path

from workflow import config


NAME = "classify"

KINDS = ("yes", "no")


def inputs(ctx) -> list[Path]:
    return [ctx.artifact("p2", kind) for kind in KINDS]


def outputs(ctx) -> list[Path]:
    return [config.classified(ctx.root, kind) for kind in KINDS]


def is_done(ctx) -> bool:
    # Global across every bundle, like the phase done-set: their existence says
    # nothing about this bundle, and union is idempotent -- but see merge, whose
    # answer this mirrors. `archive` relocates this fold's inputs into
    # p2/done/out, so an input's absence is the record that its fold ran.
    # Tested directly rather than via bundle.has_source: `extract` places both
    # artifacts unconditionally, so at this point in the recipe they are
    # present unless archive has taken them.
    return not any(ctx.artifact("p2", kind).exists() for kind in KINDS)


def run_step(ctx) -> None:
    # Per kind, because archive moves them one at a time: a crash between the
    # two leaves one gone, and re-running must fold what is left rather than
    # reach for what is not.
    for kind in KINDS:
        source = ctx.artifact("p2", kind)
        if source.exists():
            config.fold_classified(ctx.root, kind, source)
