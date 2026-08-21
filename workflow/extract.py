# extract.py
#
# Read an archived result corpus without disturbing the queue.
#
# The first recipe. This command used to carry its own resolver and its own
# extraction routine; both are gone. What is left is a STEPS list and an
# argument parser -- resolution is `select`, filtering is `filter_results`, and
# emitting a set is the standard producing-op convention.

import argparse

from pathlib import Path

from workflow import command, context, log, steps, usage
from workflow.steps import filter as filter_step


STEPS = [filter_step]


class ExtractYes(command.Action):
    def __init__(self):
        super().__init__(
            summary="yes     — extract YES pairs from p1/done/out without queueing",
            positional="all|JSONL-FILE",
        )

    def parser(self):
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--pm", "--prob-min", dest="prob_min", type=float,
                       default=0.9, metavar="PMIN",
                       help="minimum YES probability (default: 0.9)")
        p.add_argument("--pr", "--prob-range", dest="prob_range", type=float,
                       default=0.1, metavar="PRANGE",
                       help="probability range width (default: 0.1)")
        p.add_argument("-o", "--output", type=Path, required=True, metavar="FILE",
                       help="write pairs to FILE")
        return p

    def run(self, command, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command))
        if len(rest) > 1:
            return usage.invalid_argument(rest[1], self.format_help(command))

        ctx = context.Context(root=opts.dir, phase="p1", force=opts.force,
                              selector=rest[0], dest=opts.output,
                              pmin=opts.prob_min, prange=opts.prob_range)

        code = steps.run_steps(STEPS, ctx)
        if code == 0:
            log.success(f"YES pairs at {ctx.dest}")
        return code


P1_YES = ExtractYes()
