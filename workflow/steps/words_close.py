"""Discard non-authoritative active state after dictionary publication."""

import shutil

from workflow import bundle
from workflow.steps import p2_retrieve, words_extract


NAME = "close"


def inputs(ctx):
    return []


def outputs(ctx):
    return []


def run_step(ctx) -> None:
    # The source goes first. Nothing durable records that publication happened,
    # so its absence is the whole of that record -- see the run_steps call in
    # complete.CompleteWords.run. Unlinking it ahead of the derivatives means a
    # failure below leaves stray files rather than a bundle a rerun would
    # publish a second time.
    source = bundle.source(ctx)
    derived = bundle.filtered(source)
    source.unlink()
    for path in words_extract.outputs(ctx):
        path.unlink(missing_ok=True)
    derived.unlink(missing_ok=True)
    shutil.rmtree(p2_retrieve.partial_dir(ctx), ignore_errors=True)
    bundle.finish(ctx)
