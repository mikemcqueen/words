# review.py
#
# `submit` then `eval` in one step. Nothing of its own: the pairs file goes to
# the phase's submit, and the flags go to the phase's eval, which owns them.

from pathlib import Path

from workflow import (
    command, config, log, names, submit, usage, eval as evaluate,
)


class Review(command.Action):
    def __init__(self, phase: str, submitter, evaluator):
        super().__init__(
            summary=f"{phase}      — submit and evaluate a pairs file",
            positional="PAIRS-FILE")
        self.phase = phase
        self.submitter = submitter
        self.evaluator = evaluator

    def parser(self):
        return self.evaluator.parser()

    def run(self, command, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command))
        if len(rest) > 1:
            return usage.invalid_argument(rest[1], self.format_help(command))
        if opts.dry_run:
            raise ValueError("--dry-run is not valid for review")

        src = rest[0]
        # eval parses its flags again from the argument list, so the pairs file
        # is swapped for its queued name in place. A second copy of the same
        # text could be a flag's value, and then which one is the file is not
        # knowable from the list.
        if argv.count(src) > 1:
            raise ValueError(f"{src} is given more than once; "
                             f"use `wf submit {self.phase}` and "
                             f"`wf eval {self.phase}`")
        self.evaluator.check(opts)

        code = self.submitter.run(f"submit {self.phase}", opts, [src])
        if code != 0:
            return code

        queued = names.queue_name(self.phase, Path(src).name)
        eval_argv = [queued if arg == src else arg for arg in argv]
        try:
            code = self.evaluator.run(f"eval {self.phase}", opts, eval_argv)
        except Exception:
            self._hint(opts, queued)
            raise
        if code != 0:
            self._hint(opts, queued)
        return code

    def _hint(self, opts, queued: str) -> None:
        # Only while eval failed before opening the bundle. Once the file has
        # moved out of the queue, eval's own error is the whole story.
        if (config.path(opts.dir, [self.phase, "queued"]) / queued).is_file():
            log.warn(f"{queued} is still queued; continue with "
                     f"`wf eval {self.phase} {queued}`")


P1 = Review("p1", submit.P1, evaluate.P1)
P2 = Review("p2", submit.P2, evaluate.P2)
