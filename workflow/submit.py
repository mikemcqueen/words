# submit.py
#
# Place a file into a phase's queue, sorted and deduped.
#
# p1 and p2 differ in three things -- the phase, the word used in help and log
# lines, and the name of the positional. They are one class and a two-row
# table, not two files. What the queued copy is *called* is not among the
# three: that is the phase's queue contract, and it lives in names.py where
# `eval` reads the same table.

from pathlib import Path

from workflow import command, config, fs, log, names, setops


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
        log.success(f"Submitted {self.label} {src.name}")
        return 0


P1 = Submit(phase="p1", label="pairs",           positional="PAIRS-FILE")
P2 = Submit(phase="p2", label="review-candidate", positional="PAIRS-FILE")
