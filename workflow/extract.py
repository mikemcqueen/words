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

from workflow import command, config, context, fs, log, steps, usage
from workflow.steps import filter as filter_step


STEPS = [filter_step]


class ExtractYes(command.Action):
    def __init__(self):
        super().__init__(
            summary="yes     — extract YES pairs from p1/done/out without queueing",
            positional="[all|JSONL-FILE]",
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
        p.add_argument("--pairs", type=Path, metavar="PAIRS-FILE",
                       help="scan all results and restrict output to pairs in FILE")
        p.add_argument("--results-dir", type=Path, metavar="RESULTS-DIR",
                       help="with --pairs, scan DIR instead of p1/done/out")
        return p

    def run(self, command, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if opts.results_dir is not None and opts.pairs is None:
            log.error("--results-dir requires --pairs")
            return 2

        if opts.pairs is None and not rest:
            return usage.missing_argument(self.format_help(command))
        if opts.pairs is not None:
            if rest:
                return usage.invalid_argument(rest[0], self.format_help(command))
            fs.raise_if_not_file(opts.pairs)
            results_dir = opts.results_dir or config.path(
                opts.dir, ["p1", "done", "out"])
            fs.raise_if_not_dir(results_dir)
            selector = "all"
        else:
            if len(rest) > 1:
                return usage.invalid_argument(rest[1], self.format_help(command))
            results_dir = None
            selector = rest[0]

        ctx = context.Context(root=opts.dir, phase="p1", force=opts.force,
                              selector=selector, dest=opts.output,
                              results_dir=results_dir, pairs_path=opts.pairs,
                              pmin=opts.prob_min, prange=opts.prob_range)

        code = steps.run_steps(STEPS, ctx)
        if code == 0:
            log.success(f"YES pairs at {ctx.dest}")
        return code


P1_YES = ExtractYes()
