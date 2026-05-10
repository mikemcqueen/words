# eval_pairs.py

from pathlib import Path
from plumbum.cmd import comm

from workflow import log, fs, config, usage

def _usage_text():
    return "usage: wf eval pairs PAIRS-FILE"


def help_summary(name):
    return "pairs   — evaluate pairs"


def show_help(command, opts, argv):
    return usage.default_help(help_summary(command), argv, _usage_text())


def move_pairs_done(src_pairs: Path, phase: str, opts) -> Path:
    dst_dir = config.path(opts.dir, [phase, "eval"])
    dst_pairs = dst_dir / src_pairs.name
    if not opts.force:
        fs.raise_if_exists(dst_pairs)

    done_pairs = config.path(opts.dir, [phase, "done"]) / f"{phase}_done"
    if done_pairs.is_file():
        (comm["-23", str(src_pairs), str(done_pairs)] > str(dst_pairs))()
        src_pairs.unlink()
    else:
        src_pairs.rename(dst_pairs)

    return dst_pairs


def _eval_pairs(src_pairs: Path, opts) -> Path:
    return move_pairs_done(src_pairs, "p1", opts)


def run(command, opts, argv):
    if not argv:
        details = _usage_text()
        return usage.missing_argument(details)

    src_dir = config.path(opts.dir, ["p1", "queued"]);
    src_pairs = src_dir / argv[0]
    fs.raise_if_not_file(src_pairs)

    dst_pairs = _eval_pairs(src_pairs, opts)
    if not dst_pairs:
        return 1
    
    # TODO: (optionally?) copy file to somewhere specified by user

    log.success(f"Ready for evalpairs: {src_pairs.name}")
    return 0
