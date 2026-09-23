# submit.py
#
# Place a file into a phase's queue, sorted and deduped.
#
# p1 and p2 share the queue operation. P2 additionally validates the pair
# syntax at this outside boundary: everything downstream treats a line as an
# opaque set member, so accepting counted or otherwise malformed rows here
# would keep them from matching the classified pair sets.

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

    def parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--as", dest="as_name", metavar="NAME",
                            help="queue name, in place of the file's own "
                                 "(the queue suffix is still added)")
        return parser

    def prepare(self, src: Path) -> Path:
        return src

    def run(self, command, opts, argv) -> int:
        src = _resolve_input(self.parse(opts, argv))
        chosen = opts.as_name if opts.as_name is not None else src.name
        dst = (config.path(opts.dir, [self.phase, "queued"])
               / names.queue_name(self.phase, chosen))
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
