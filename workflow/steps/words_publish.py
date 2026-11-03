"""Publish a validated dictionary review as one managed dictionary round."""

from pathlib import Path

from workflow import dictionary, fs, log
from workflow.steps import p2_retrieve, words_extract


NAME = "publish"


def inputs(ctx) -> list[Path]:
    return words_extract.outputs(ctx) + [p2_retrieve.enex_dir(ctx)]


def outputs(ctx) -> list[Path]:
    return []


def run_step(ctx, force_verdicts: bool = False) -> None:
    for path in inputs(ctx):
        if path == p2_retrieve.enex_dir(ctx):
            fs.raise_if_not_dir(path)
        else:
            fs.raise_if_not_file(path)
    result = dictionary.publish_words(
        ctx.root, ctx.bundle_name, words_extract.artifact(ctx, "input.words"),
        yes=words_extract.artifact(ctx, "yes"),
        no=words_extract.artifact(ctx, "no"),
        force_verdicts=force_verdicts,
        retry=(f"correct the notes in the note application, then run "
               f"`wf -f complete words {ctx.bundle_name}`"),
        enex=p2_retrieve.enex_dir(ctx))
    undecided = result.reviewed - result.yes - result.no
    log.success(f"{result.reviewed} words reviewed: {result.yes} YES, "
                f"{result.no} NO, {undecided} undecided")
    dictionary.report_publication(ctx.root, result)
