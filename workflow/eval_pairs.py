# eval_pairs.py

import argparse
from pathlib import Path
from plumbum.cmd import comm

from workflow import log, fs, config, usage

def help_summary(name):
    return "pairs   — evaluate pairs"


def _format_help(command, opts, argv):
    return usage.format_help(command, help_summary(command), _make_parser(), "PAIRS-FILE")


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


def make_dst_pairs_path(src_pairs: Path, phase: str, opts) -> Path:
    dst_dir = config.path(opts.dir, [phase, "eval"])
    dst_pairs = dst_dir / src_pairs.name
    if not opts.force:
        fs.raise_if_exists(dst_pairs)
    return dst_pairs


def filter_done_pairs(src_pairs: Path, dst_pairs: Path, phase: str, opts) -> None:
    done_pairs = config.path(opts.dir, [phase, "done"]) / f"{phase}_done"
    if done_pairs.is_file():
        (comm["-23", str(src_pairs), str(done_pairs)] > str(dst_pairs))()
        src_pairs.unlink()
    else:
        src_pairs.rename(dst_pairs)


def _eval_pairs(src_pairs: Path, opts) -> Path:
    dst_pairs = make_dst_pairs_path(src_pairs, "p1", opts)
    if not opts.no_filter:
        filter_done_pairs(src_pairs, dst_pairs, "p1", opts)
    else:
        src_pairs.rename(dst_pairs)
        log.info("Skipped done pair filtering")
    return dst_pairs


def run(command, opts, argv):
    argv, opts = _parse_args(argv, opts)
    if not argv:
        return usage.missing_argument(_format_help(command, opts, argv))

    src_dir = config.path(opts.dir, ["p1", "queued"]);
    src_pairs = src_dir / argv[0]
    fs.raise_if_not_file(src_pairs)

    dst_pairs = _eval_pairs(src_pairs, opts)
    if not dst_pairs:
        return 1
    
    # TODO: (optionally?) copy file to somewhere specified by user

    log.success(f"{fs.line_count(dst_pairs)} pairs ready for evalpairs: {src_pairs.name}")
    return 0
