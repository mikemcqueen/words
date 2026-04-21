from pathlib import Path
from plumbum.cmd import comm
from workflow import log, config


def help_summary(name):
    return "pairs   — move a queued pairs file to the running stage"


def help(command, opts, argv):
    print(help_summary(command))
    print("usage: wf eval pairs [pair-file]")
    return 0


def _phase1_dir(opts) -> Path | None:
    d = opts.dir / config.ROOT / "p1"
    if not d.is_dir():
        log.error(f"{d} does not exist; run `wf init` first")
        return None
    return d


def _resolve_pair_file(qdir: Path, argv) -> Path | None:
    if argv:
        name = argv[0]
        src = qdir / Path(name).name
        if not src.is_file():
            log.error(f"not in queue: {src}")
            return None
        return src
    # AI - do better.
    files = [p for p in qdir.iterdir() if p.is_file()]
    if not files:
        log.error("The pairs queue is empty.")
        return None
    if len(files) > 1:
        log.error("Missing FILE parameter. Use `wf show pairs` to see files.")
        return None
    return files[0]


def run(command, opts, argv):
    p1 = _phase1_dir(opts)
    if p1 is None:
        return 1

    qdir = p1 / "queued"
    if not qdir.is_dir():
        log.error(f"{qdir} does not exist; run `wf init` first")
        return 1

    src = _resolve_pair_file(qdir, argv)
    if src is None:
        return 2

    rdir = p1 / "running"
    if not rdir.is_dir():
        log.error(f"{rdir} does not exist; run `wf init` first")
        return 1

    done_pairs = p1 / "done" / "pairs"
    dst = rdir / src.name
    orig = rdir / f"{src.name}.orig"

    if dst.exists() or orig.exists():
        log.warn(f"already running: {dst}")
        return 1

    if done_pairs.is_file():
        (comm["-23", str(src), str(done_pairs)] > str(dst))()
        src.rename(orig)
    else:
        src.rename(orig)
        dst.write_bytes(orig.read_bytes())

    log.success(f"Ready for evalpairs: {dst}")
    return 0
