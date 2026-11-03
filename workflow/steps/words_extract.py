"""Extract and validate the YES/NO/undecided partition of a word review."""

import subprocess
import tempfile

from pathlib import Path

from workflow import bundle, dictionary, fs, setops
from workflow.steps import p2_retrieve


NAME = "extract"
KINDS = ("yes", "no", "input.words")


def artifact(ctx, kind: str) -> Path:
    return ctx.bundle_dir / f"{ctx.bundle_name}.{kind}"


def inputs(ctx) -> list[Path]:
    return sorted(p2_retrieve.enex_dir(ctx).glob("*.enex"))


def outputs(ctx) -> list[Path]:
    return [artifact(ctx, kind) for kind in KINDS]


def _normalize(source: Path, destination: Path) -> Path:
    rows = []
    for number, line in enumerate(source.read_text().splitlines(), start=1):
        word = dictionary.parse_word_row(source, number, line)
        if word is not None:
            rows.append(word)
    unsorted = destination.with_name(destination.name + ".unsorted")
    dictionary.write_words(unsorted, rows)
    try:
        return setops.merge([unsorted], destination)
    finally:
        unsorted.unlink(missing_ok=True)


def _parse(enex: list[Path], note_type: str, directory: Path) -> Path:
    parsed = []
    for index, part in enumerate(enex):
        raw = directory / f"{index}.{note_type}.raw"
        with raw.open("w") as output:
            # --two-checkboxes makes a row count only when it carries both
            # boxes, so NONE is a row with neither ticked, not a one-box row.
            subprocess.run(["note", "--parse-file", str(part), "--type",
                            note_type, "--lines", "--two-checkboxes"],
                           stdout=output, check=True)
        parsed.append(_normalize(raw, directory / f"{index}.{note_type}.words"))
    return setops.merge(parsed, directory / note_type.lower())


def validate_results(reviewed_input: Path, yes: Path, no: Path,
                     undecided: Path, bundle_name: str = "NAME") -> None:
    """Require YES, NO and undecided to partition the evaluated input."""
    reviewed = dictionary.word_set(reviewed_input)
    yes_set, no_set = dictionary.word_set(yes), dictionary.word_set(no)
    undecided_set = dictionary.word_set(undecided)
    conflicting, extra_verdicts = dictionary.verdict_errors(reviewed, yes_set,
                                                            no_set)
    listed = yes_set | no_set | undecided_set
    extra = sorted(set(extra_verdicts) | (undecided_set - reviewed))
    missing = sorted(reviewed - listed)
    if not conflicting and not extra and not missing:
        return

    sections = []
    for label, words in (("ticked both YES and NO", conflicting),
                         ("ticked but not in the evaluated input", extra),
                         ("missing from the notes", missing)):
        if words:
            sections.append(f"{label}:\n{dictionary.listing(words)}")
    raise ValueError(
        f"invalid dictionary review partition\n"
        f"input {len(reviewed):,}, YES {len(yes_set):,}, NO {len(no_set):,}, "
        f"undecided {len(undecided_set):,}; conflicting {len(conflicting):,}, "
        f"extra {len(extra):,}, missing {len(missing):,}\n"
        + "\n".join(sections) + "\n"
        f"Correct the review notes in the note application, then run "
        f"`wf -f complete words {bundle_name}`.")


def run_step(ctx) -> None:
    fs.raise_if_not_dir(p2_retrieve.enex_dir(ctx))
    enex = inputs(ctx)
    if not enex:
        raise ValueError(f"no ENEX parts in {p2_retrieve.enex_dir(ctx)}")
    with tempfile.TemporaryDirectory(prefix="wf-words-extract-") as tmp:
        scratch = Path(tmp)
        reviewed_input = _normalize(bundle.evaluated(ctx),
                                    scratch / "input.words")
        yes = _parse(enex, "YES", scratch)
        no = _parse(enex, "NO", scratch)
        undecided = _parse(enex, "NONE", scratch)
        validate_results(reviewed_input, yes, no, undecided, ctx.bundle_name)

        staged = {
            "yes": yes,
            "no": no,
            "input.words": reviewed_input,
        }
        for kind, path in staged.items():
            destination = artifact(ctx, kind)
            if ctx.force:
                fs.optional_file(destination)
            elif destination.exists() or destination.is_symlink():
                raise fs.file_already_exists_error(destination)
        for kind, path in staged.items():
            setops.place(setops.Staged(path, artifact(ctx, kind), True))
