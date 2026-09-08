"""Extract and validate the checked/unchecked partition of a word review."""

import filecmp
import subprocess
import tempfile

from pathlib import Path

from workflow import bundle, dictionary, fs, setops
from workflow.steps import p2_retrieve


NAME = "extract"
KINDS = ("yes", "no", "all", "input.words")


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
    unsorted.write_text("".join(f"{word}\n" for word in rows))
    try:
        return setops.merge([unsorted], destination)
    finally:
        unsorted.unlink(missing_ok=True)


def _parse(enex: list[Path], note_type: str, directory: Path) -> Path:
    parsed = []
    for index, part in enumerate(enex):
        raw = directory / f"{index}.{note_type}.raw"
        with raw.open("w") as output:
            subprocess.run(["note", "--parse-file", str(part), "--type",
                            note_type, "--lines"], stdout=output, check=True)
        parsed.append(_normalize(raw, directory / f"{index}.{note_type}.words"))
    return setops.merge(parsed, directory / note_type.lower())


def validate_results(reviewed_input: Path, yes: Path, no: Path,
                     all_words: Path, overlap: Path,
                     bundle_name: str = "NAME") -> None:
    """Require YES and NO to be an exact disjoint partition of input."""
    setops.merge([yes, no], all_words)
    setops.common(yes, no, overlap)
    same_input = filecmp.cmp(all_words, reviewed_input, shallow=False)
    has_overlap = fs.line_count(overlap) != 0
    if same_input and not has_overlap:
        return

    directory = all_words.parent
    missing = setops.diff(reviewed_input, all_words, directory / "missing")
    extra = setops.diff(all_words, reviewed_input, directory / "extra")
    failures = []
    if not same_input:
        failures.append("YES/NO union differs from the evaluated input")
    if has_overlap:
        failures.append("words are marked both checked and unchecked")
    conflicts = overlap.read_text().splitlines()
    shown = ""
    if conflicts:
        listed = "\n  ".join(conflicts[:20])
        remainder = len(conflicts) - 20
        more = f"\n  ... and {remainder:,} more" if remainder > 0 else ""
        shown = f"\nConflicting words:\n  {listed}{more}"
    raise ValueError(
        f"invalid dictionary review partition: {'; '.join(failures)}\n"
        f"input {fs.line_count(reviewed_input):,}, "
        f"YES {fs.line_count(yes):,}, NO {fs.line_count(no):,}; "
        f"missing {fs.line_count(missing):,}, extra {fs.line_count(extra):,}, "
        f"conflicting {fs.line_count(overlap):,}.{shown}\n"
        f"Correct the checkbox review notes in the note application, then run "
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
        no = _parse(enex, "NONE", scratch)
        all_words = scratch / "all"
        overlap = scratch / "both"
        validate_results(reviewed_input, yes, no, all_words, overlap,
                         ctx.bundle_name)

        staged = {
            "yes": yes,
            "no": no,
            "all": all_words,
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
