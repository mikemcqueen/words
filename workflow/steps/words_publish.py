"""Publish a validated dictionary review as one managed dictionary round."""

from pathlib import Path

from workflow import dictionary, fs, log
from workflow.steps import p2_retrieve, words_extract


NAME = "publish"


def inputs(ctx) -> list[Path]:
    return words_extract.outputs(ctx) + [p2_retrieve.enex_dir(ctx)]


def outputs(ctx) -> list[Path]:
    return []


def run_step(ctx) -> None:
    for path in inputs(ctx):
        if path == p2_retrieve.enex_dir(ctx):
            fs.raise_if_not_dir(path)
        else:
            fs.raise_if_not_file(path)
    reviewed = words_extract.artifact(ctx, "input.words")
    yes = words_extract.artifact(ctx, "yes")
    no = words_extract.artifact(ctx, "no")
    result = dictionary.publish_words(ctx.root, ctx.bundle_name, reviewed, no,
                                      p2_retrieve.enex_dir(ctx))
    log.success(f"{fs.line_count(reviewed)} words reviewed: "
                f"{fs.line_count(yes)} checked, {fs.line_count(no)} unchecked")
    dictionary.report_publication(ctx.root, result)
