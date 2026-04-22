
from pathlib import Path
from plumbum.cmd import sort
from workflow import config, fs, log


def help_summary(name):
    return "pairs   — submit a pairs file into p1/queued (sorted, deduped)"


# TODO: config.path
def _queued_dir(opts) -> Path:
    d = opts.dir / config.ROOT / "p1" / "queued"
    fs.raise_if_not_dir(d)
    return d


def _resolve_input(argv) -> Path:
    if not argv:
        log.error("submit pairs: missing <pair-file> argument")
        return None
    src = Path(argv[0]).resolve()
    fs.raise_if_not_file(src)
    return src


def _sort_unique(src: Path, dst: Path):
    (sort["-u", str(src)] > str(dst))()


def run(command, opts, argv):
    src = _resolve_input(argv)
    dst = _queued_dir(opts) / src.name
    if dst.exists():
        log.warn(f"Already queued: {dst}")
        return 2
    _sort_unique(src, dst)
    log.success(f"Queued {dst}")
    return 0


def help(command, opts, argv):
    print(help_summary(command))
    print("usage: wf submit pairs <pair-file>")
    return 0
