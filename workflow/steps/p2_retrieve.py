# steps/p2_retrieve.py
#
# Fetch the manually-classified note parts back out of the note store.
#
# This is the one step whose output set is not knowable in advance: parts are
# probed .aa, .ab, ... until one is missing, so given an enex/ holding three
# files the filesystem cannot say whether that is "finished at three" or
# "crashed after three". The fix is to make the *directory* the atomic unit:
# parts land in enex.part/, and only the 404 -- the moment completion is
# actually known -- renames it to enex/. So enex/ exists iff retrieve finished,
# is_done is one stat, and a partial run resumes into the parts already there
# instead of raising. Each part is placed atomically too, so a truncated .enex
# can never pass for a complete one.

import subprocess

from pathlib import Path

from workflow import batch, fs, log


NAME = "retrieve"

MAX_NOTE_PARTS = 26


def enex_dir(ctx) -> Path:
    return ctx.batch_dir / "enex"


def partial_dir(ctx) -> Path:
    return ctx.batch_dir / "enex.part"


def inputs(ctx) -> list[Path]:
    return [batch.evaluated(ctx)]


def outputs(ctx) -> list[Path]:
    return [enex_dir(ctx)]


def is_done(ctx) -> bool:
    return enex_dir(ctx).is_dir()


def _title(source: Path, index: int) -> str:
    return f"{source.name}.a{chr(ord('a') + index)}"


def run_step(ctx) -> None:
    source = batch.evaluated(ctx)
    staging = partial_dir(ctx)
    staging.mkdir(exist_ok=True)

    fetched = 0
    for index in range(MAX_NOTE_PARTS):
        title = _title(source, index)
        part = staging / f"{title}.enex"
        if part.exists():
            fetched += 1
            continue

        result = subprocess.run(["note", "-pf.72", "--get", title, "--production"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            if "note not found" in result.stderr:
                break
            raise RuntimeError(f"failed to retrieve note {title}:\n{result.stderr}")

        tmp = staging / f"{title}.enex.tmp"
        tmp.write_text(result.stdout)
        tmp.replace(part)
        fetched += 1

    if not fetched:
        raise RuntimeError(f"no note parts found for {source.name} (expected at least .aa)")

    log.info(f"retrieved {fetched} note part(s)")
    # `-f` skips is_done, so this rename can land on an enex/ a previous run
    # already completed. Tolerate that rather than dying on a non-empty target.
    fs.rename_once(staging, enex_dir(ctx), ctx.force)
