from pathlib import Path
from plumbum.cmd import sort
from workflow import log, config


def help_summary(name):
    return "pairs   — submit a pairs file into p1/queued (sorted, deduped)"


def _queued_dir(opts) -> Path | None:
    d = opts.dir / config.ROOT / "p1" / "queued"
    if not d.is_dir():
        log.error(f"{d} does not exist; run `wf init` first")
        return None
    return d


def _resolve_input(argv) -> Path | None:
    if not argv:
        log.error("submit pairs: missing <pair-file> argument")
        return None
    src = Path(argv[0]).resolve()
    if not src.is_file():
        log.error(f"not a file: {src}")
        return None
    return src


def _sort_unique(src: Path, dst: Path) -> bool:
    (sort["-u", str(src)] > str(dst))()
    return True


def run(command, opts, argv):
    src = _resolve_input(argv)
    if src is None:
        return 2
    qdir = _queued_dir(opts)
    if qdir is None:
        return 1
    dst = qdir / src.name
    if dst.exists():
        log.warn(f"already queued: {dst}")
        return 1
    _sort_unique(src, dst)
    log.success(f"Queued {dst}")
    return 0


def help(command, opts, argv):
    print(help_summary(command))
    print("usage: wf submit pairs <pair-file>")
    return 0
