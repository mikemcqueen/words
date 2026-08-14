# extract_p1_yes.py

import argparse
import subprocess
import tempfile

from pathlib import Path

from src.filter import filter_results
from workflow import config, fs, log, usage


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


def _result_paths(opts, source: str) -> list[Path]:
    results_dir = config.path(opts.dir, ["p1", "done", "out"])
    fs.raise_if_not_dir(results_dir)

    if source != "all":
        path = results_dir / source
        fs.raise_if_not_file(path)
        return [path]

    paths = [p for p in results_dir.glob("*.jsonl") if p.is_file()]
    if not paths:
        raise ValueError(f"no .jsonl files in {results_dir}")
    return paths


def _extract(paths: list[Path], pmin: float, prange: float,
             output: Path, force: bool) -> int:
    fs.raise_if_not_dir(output.parent)
    if not force:
        fs.raise_if_exists(output)

    with tempfile.NamedTemporaryFile(mode="w", prefix="wf-extract-p1-yes-",
                                     suffix=".pairs") as matches:
        for path in paths:
            filter_results(str(path), True, matches, pmin=pmin,
                           prng=prange, use_max=False)
        matches.flush()

        sort_args = ["sort", "-u", matches.name]
        with output.open("w") as f:
            subprocess.run(sort_args, stdout=f, check=True)

    log.info(f"Extracted YES pairs from {len(paths)} p1 result files")
    return 0


def run(command, opts, argv):
    local_opts, rest = _make_local_parser().parse_known_args(argv)
    if not rest:
        return usage.missing_argument(_format_help(command))
    if len(rest) > 1:
        return usage.invalid_argument(rest[1], _format_help(command))

    paths = _result_paths(opts, rest[0])
    return _extract(paths, local_opts.prob_min, local_opts.prob_range,
                    local_opts.output, opts.force)
