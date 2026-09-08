# submit.py
#
# Place a file into a phase's queue, sorted and deduped.
#
# p1 and p2 differ in three things -- the phase, the word used in help and log
# lines, and the name of the positional. They are one class and a two-row
# table, not two files. What the queued copy is *called* is not among the
# three: that is the phase's queue contract, and it lives in names.py where
# `eval` reads the same table.

import argparse
import tempfile

from pathlib import Path

from workflow import command, config, dictionary, fs, log, names, setops, usage


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

    def run(self, command, opts, argv) -> int:
        src = _resolve_input(argv)
        dst = (config.path(opts.dir, [self.phase, "queued"])
               / names.queue_name(self.phase, src.name))
        if not opts.force:
            fs.raise_if_exists(dst)

        setops.merge([src], dst)
        # The queued name, not the one handed in: the queue contract may have
        # stamped a suffix on, and this is the name `eval` will want.
        log.success(f"Submitted {self.label} {dst.name}")
        return 0


P1 = Submit(phase="p1", label="pairs",           positional="PAIRS-FILE")
P2 = Submit(phase="p2", label="review-candidate", positional="PAIRS-FILE")


class SubmitWords(command.Action):
    """Queue ranked dictionary text without changing a byte."""

    def __init__(self):
        super().__init__(summary="words   — submit dictionary words for review",
                         positional="FILE")

    def parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--as", dest="as_name", metavar="NAME",
                            help="queue and archive name")
        return parser

    def run(self, command_text, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command_text))
        if len(rest) > 1:
            return usage.invalid_argument(rest[1],
                                          self.format_help(command_text))

        src = Path(rest[0])
        if src.is_symlink():
            raise ValueError(f"symlink input not allowed: {src}")
        fs.raise_if_not_readable(src)

        chosen = opts.as_name if opts.as_name is not None else src.name
        names.check_name(chosen, "submitted filename")
        dictionary.archive_name(chosen)
        dst = config.path(opts.dir, ["dict", "queued"]) / chosen
        if opts.force:
            fs.optional_file(dst)
        elif dst.exists() or dst.is_symlink():
            raise fs.file_already_exists_error(dst)

        prepared = None
        try:
            with tempfile.NamedTemporaryFile(
                    prefix="wf-submit-words-", delete=False) as tmp:
                prepared = Path(tmp.name)
                with src.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        tmp.write(block)
            setops.place(setops.Staged(prepared, dst, replaced=True))
            prepared = None
        finally:
            if prepared is not None:
                prepared.unlink(missing_ok=True)

        log.success(f"Submitted words {dst.name}")
        return 0


WORDS = SubmitWords()
