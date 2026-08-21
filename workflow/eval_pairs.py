# eval_pairs.py

import argparse
from pathlib import Path

from workflow import log, fs, batch, config, setops, usage

def help_summary(name):
    return "p1      — evaluate pairs"


def _format_help(command, opts, argv):
    return usage.format_help(command, help_summary(command), _make_parser(), "SLUG")


def show_help(command, opts, argv):
    text = _format_help(command, opts, argv)
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0


def add_no_filter_flag(parser):
    parser.add_argument("--no-filter", action="store_true",
                        help="skip filtering already-evaluated pairs")


def _make_parser():
    p = argparse.ArgumentParser(add_help=False)
    add_no_filter_flag(p)
    return p


def _parse_args(argv, opts):
    local_opts, rest = _make_parser().parse_known_args(argv)
    vars(opts).update(vars(local_opts))
    return rest, opts


def make_done_pairs_path(phase: str, opts) -> Path:
    return config.path(opts.dir, [phase, "done"]) / f"{phase}_done.pairs"


def make_filtered_pairs_path(src_pairs: Path) -> Path:
    return src_pairs.with_name(src_pairs.name + ".filtered")


def filter_done_pairs(src_pairs: Path, phase: str, opts) -> None:
    done_pairs = make_done_pairs_path(phase, opts)
    if not done_pairs.is_file():
        return src_pairs

    filtered_pairs = make_filtered_pairs_path(src_pairs)
    if not opts.force:
        fs.raise_if_exists(filtered_pairs)

    setops.diff(src_pairs, done_pairs, filtered_pairs)
    log.info(f"{fs.line_count(filtered_pairs)} filtered pairs")
    return filtered_pairs


def _eval_pairs(slug: str, opts) -> Path:
    dst_pairs = batch.begin(opts, "p1", slug)
    log.info(f"{fs.line_count(dst_pairs)} source pairs")
    if not opts.no_filter:
        dst_pairs = filter_done_pairs(dst_pairs, "p1", opts)
    return dst_pairs


def run(command, opts, argv):
    argv, opts = _parse_args(argv, opts)
    if not argv:
        return usage.missing_argument(_format_help(command, opts, argv))

    # The positional is the batch directory name, and the queued artifact is
    # found under it by prefix.
    dst_pairs = _eval_pairs(argv[0], opts)
    if not dst_pairs:
        return 1
    
    # TODO: (optionally?) copy file to somewhere specified by user

    log.success(f"{fs.line_count(dst_pairs)} pairs ready for evalpairs: {dst_pairs.name}")
    return 0
