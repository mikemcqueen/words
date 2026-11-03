# submit.py
#
# Place a file into a phase's queue, sorted and deduped.
#
# p1 and p2 share the queue operation. P2 additionally validates the pair
# syntax at this outside boundary: everything downstream treats a line as an
# opaque set member, so accepting counted or otherwise malformed rows here
# would keep them from matching the classified pair sets.

import argparse
import re
import tempfile

from pathlib import Path

from workflow import (bundle, command, config, context, dictionary, fs, log,
                      names, notes, setops, usage)


def _resolve_input(argv) -> Path:
    if not argv:
        raise ValueError("Missing FILE parameter.")

    src = Path(argv[0]).resolve()
    fs.raise_if_not_file(src)
    return src


class Submit(command.Action):
    def __init__(self, phase: str, label: str, positional: str):
        super().__init__(
            summary=f"{phase}      — submit a {label} file into {phase}/queued "
                    f"(sorted, deduped)",
            positional=positional,
        )
        self.phase = phase
        self.label = label

    def parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--as", dest="as_name", metavar="NAME",
                            help="queue name, in place of the file's own "
                                 "(the queue suffix is still added)")
        return parser

    def check_queueable(self, opts, dst: Path) -> None:
        pass

    def prepare(self, src: Path) -> Path:
        return src

    def run(self, command, opts, argv) -> int:
        src = _resolve_input(self.parse(opts, argv))
        chosen = opts.as_name if opts.as_name is not None else src.name
        dst = (config.path(opts.dir, [self.phase, "queued"])
               / names.queue_name(self.phase, chosen))
        self.check_queueable(opts, dst)
        if not opts.force:
            fs.raise_if_exists(dst)

        prepared = self.prepare(src)
        try:
            setops.merge([prepared], dst)
            # The queued name, not the one handed in: the queue contract may have
            # stamped a suffix on, and this is the name `eval` will want.
            log.success(f"Submitted {self.label} {dst.name}")
            return 0
        finally:
            if prepared != src:
                prepared.unlink(missing_ok=True)


class SubmitP2(Submit):
    def check_queueable(self, opts, dst: Path) -> None:
        stem = names.queue_stem(self.phase, dst.name)
        bundle.check_queueable(
            context.Context(root=opts.dir, phase=self.phase, bundle_name=stem))

    def prepare(self, src: Path) -> Path:
        prepared = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", prefix="wf-submit-p2-", delete=False) as tmp:
                prepared = Path(tmp.name)
                with src.open() as source:
                    for line_number, raw in enumerate(source, 1):
                        pair = raw.rstrip("\n")
                        if not pair.strip():
                            continue
                        fields = pair.split(",")
                        if (len(fields) != 2 or
                                any(not field or field != field.strip()
                                    for field in fields)):
                            raise ValueError(
                                f"invalid pair in {src} at line {line_number}: "
                                f"expected exactly two nonempty comma-separated "
                                f"fields with no surrounding whitespace; got "
                                f"{pair!r}")
                        tmp.write(f"{pair}\n")
            return prepared
        except BaseException:
            if prepared is not None:
                prepared.unlink(missing_ok=True)
            raise


P1 = Submit(phase="p1", label="pairs",           positional="PAIRS-FILE")
P2 = SubmitP2(phase="p2", label="review-candidate", positional="PAIRS-FILE")


def separate_parts(src: Path, classified: list[Path],
                   size: int) -> list[str]:
    """Cut src into contiguous, unchanged slices of at most `size` words left
    unreviewed.

    The cut points are where eval's filter would find the next unclassified
    word past `size`; rows it would drop -- classified words, blank lines --
    stay in whichever slice they fall in, so the slices concatenate back to
    src unchanged, in its ranked order. Rows after the last unclassified word
    stay in the last slice, so no slice is left with nothing to review.
    """
    known = dictionary.classified_set(classified)
    parts: list[list[str]] = [[]]
    count = surviving = 0
    for line, _, kept in dictionary.unclassified_rows(src, known):
        if kept and count == size:
            parts.append([])
            count = 0
        parts[-1].append(line)
        if kept:
            count += 1
            surviving += 1
    if surviving == 0:
        raise ValueError(f"no unclassified words in {src}")
    return ["".join(part) for part in parts]


def separate_siblings(root: Path, chosen: str) -> list[Path]:
    """Queued files and open reviews named `chosen` or `chosen.<digits>`."""
    pattern = re.compile(re.escape(chosen) + r"(\.[0-9]+)?")
    found = []
    for directory in (config.path(root, ["dict", "queued"]),
                      config.path(root, ["dict", "eval"])):
        found += sorted(p for p in directory.iterdir()
                        if pattern.fullmatch(p.name))
    return found


def separate_names(chosen: str, count: int) -> list[str]:
    """`chosen` alone for one part, else chosen.1.. zero-padded to count."""
    if count == 1:
        return [chosen]
    width = len(str(count))
    return [f"{chosen}.{index:0{width}d}" for index in range(1, count + 1)]


class SubmitWords(command.Action):
    """Queue ranked dictionary text without changing a byte."""

    def __init__(self):
        super().__init__(summary="words   — submit dictionary words for review",
                         positional="FILE")

    def parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--as", dest="as_name", metavar="NAME",
                            help="queue and archive name")
        add_separate(parser)
        return parser

    def run(self, command_text, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command_text))
        if len(rest) > 1:
            return usage.invalid_argument(rest[1],
                                          self.format_help(command_text))
        self.queue(opts, Path(rest[0]), opts.as_name,
                   getattr(opts, "separate", False))
        return 0

    def queue(self, opts, src: Path, as_name: str | None,
              separate: bool) -> list[str]:
        """Queue src, or its `--separate` parts; return the queued names."""
        if src.is_symlink():
            raise ValueError(f"symlink input not allowed: {src}")
        fs.raise_if_not_readable(src)

        chosen = as_name if as_name is not None else src.name
        names.check_name(chosen, "submitted filename")
        dictionary.archive_name(chosen)
        queued_dir = config.path(opts.dir, ["dict", "queued"])

        if separate:
            classified = dictionary.classified_files(opts.dir)
            parts = separate_parts(src, classified, notes.CHUNK_SIZE)
        else:
            parts = None
        queued = separate_names(chosen, len(parts) if parts else 1)

        # Every name is checked before any is written, so a refusal leaves
        # the queue as it was.
        for name in queued:
            names.check_name(name, "submitted filename")
        if separate:
            # Any earlier submission under this name -- queued, or open as a
            # review in dict/eval -- is refused, -f or not. Overwriting only
            # the names this cut produces would leave an earlier, longer or
            # differently padded cut's other parts queued beside the new ones.
            existing = separate_siblings(opts.dir, chosen)
            if existing:
                raise ValueError(
                    f"earlier submissions of {chosen} exist:\n  "
                    + "\n  ".join(str(p) for p in existing)
                    + "\nEvaluate, complete or remove them before "
                      "submitting it again with --separate")
        else:
            dst = queued_dir / chosen
            if opts.force:
                fs.optional_file(dst)
            elif dst.exists() or dst.is_symlink():
                raise fs.file_already_exists_error(dst)

        staged: list[setops.Staged] = []
        placed: list[str] = []
        try:
            for index, name in enumerate(queued):
                with tempfile.NamedTemporaryFile(
                        prefix="wf-submit-words-", delete=False) as tmp:
                    staged.append(setops.Staged(Path(tmp.name),
                                                queued_dir / name, True))
                    if parts is not None:
                        tmp.write(parts[index].encode())
                    else:
                        with src.open("rb") as source:
                            while block := source.read(1024 * 1024):
                                tmp.write(block)
            while staged:
                setops.place(staged[0])
                staged.pop(0)
                placed.append(queued[len(placed)])
        except BaseException:
            if placed:
                log.warn(f"queued before the failure, in "
                         f"{queued_dir}: {', '.join(placed)}; remove them "
                         f"before submitting {chosen} again")
            raise
        finally:
            for pending in staged:
                pending.path.unlink(missing_ok=True)

        for name in queued:
            log.success(f"Submitted words {name}")
        return queued


def add_separate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--separate", action="store_true",
        help=f"queue one file per {notes.CHUNK_SIZE} unclassified words, "
             f"suffixed .1, .2, ...")


WORDS = SubmitWords()
