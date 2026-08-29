# steps/p2_close.py
#
# Close the bundle. Last, as `advance` was -- but there is nothing left to
# advance: `classify` folded both verdicts into the standing classified sets
# and `archive` emptied the bundle into done/. p2's NO set used to be published
# into p3's queue for a second automated pass; now that the notes are parsed
# with --two-checkboxes, a NO is a reviewer's explicit verdict rather than the
# absence of a YES, and there is nothing left for another pass to decide.
#
# p1 still closes inside its own `advance`, where closing is the tail of a
# publication that really happens.

from pathlib import Path

from workflow import bundle


NAME = "close"


def inputs(ctx) -> list[Path]:
    return []


def outputs(ctx) -> list[Path]:
    return []


def is_done(ctx) -> bool:
    # The bundle directory exists iff work is in flight, so its absence is the
    # whole answer -- and it is one stat.
    return not ctx.bundle_dir.exists()


def run_step(ctx) -> None:
    bundle.finish(ctx)
