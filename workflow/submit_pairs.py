# submit_pairs.py

from pathlib import Path
from plumbum.cmd import sort

from . import fs, log, config


def help_summary(name):
    return "pairs   — submit a pairs file into p1/queued (sorted, deduped)"


def _resolve_input(argv) -> Path:
    if not argv:
        raise ValueError("Missing FILE parameter.")

    src = Path(argv[0]).resolve()
    fs.raise_if_not_file(src)
    return src


def _make_pairs_filename(fn: str) -> str:
    if not fn.endswith(".pairs"):
        return fn + ".pairs"
    return fn


def _sort_unique(src: Path, dst: Path):
    (sort["-u", str(src)] > str(dst))()


def run(command, opts, argv):
    src = _resolve_input(argv)
    dst = config.path(opts.dir, ["p1", "queued"]) / _make_pairs_filename(src.name)
    if dst.exists():
        log.warn(f"File already in queue: {dst}")
        return 2
    _sort_unique(src, dst)
    log.success(f"Queued {dst}")
    return 0


def help(command, opts, argv):
    print(help_summary(command))
    print("usage: wf submit pairs <pair-file>")
    return 0
