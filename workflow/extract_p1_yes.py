# extract_p1_yes.py
#
# The first recipe. This command used to carry its own resolver and its own
# extraction routine; both are gone. What is left is a STEPS list and an
# argument parser -- resolution is `select`, filtering is `filter_results`, and
# emitting a set is the standard producing-op convention.

import argparse

from pathlib import Path

from workflow import context, log, steps, usage
from workflow.steps import filter as filter_step


STEPS = [filter_step]


def help_summary(name):
    return "yes     — extract YES pairs from p1/done/out without queueing"


def _make_local_parser():
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


def _format_help(command):
    return usage.format_help(command, help_summary(command),
                             local_parser=_make_local_parser(),
                             positional="all|JSONL-FILE")


def show_help(command, opts, argv):
    text = _format_help(command)
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0


def run(command, opts, argv):
    local_opts, rest = _make_local_parser().parse_known_args(argv)
    if not rest:
        return usage.missing_argument(_format_help(command))
    if len(rest) > 1:
        return usage.invalid_argument(rest[1], _format_help(command))

    ctx = context.Context(root=opts.dir, phase="p1", force=opts.force,
                          selector=rest[0], dest=local_opts.output,
                          pmin=local_opts.prob_min, prange=local_opts.prob_range)

    code = steps.run_steps(STEPS, ctx)
    if code == 0:
        log.success(f"YES pairs at {ctx.dest}")
    return code
