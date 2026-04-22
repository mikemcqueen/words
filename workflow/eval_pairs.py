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


def _eval_pairs(src_dir: Path, src_pairs: Path, dst_dir: Path) -> Path:
    done_pairs = p1 / "done" / "pairs"
    dst_pairs = dst_dir / src_pairs.name
    dst_orig = dst_dir / f"{src_pairs.name}.orig"

    if dst_pairs.exists() or orig.exists():
        log.warn(f"already running: {dst_pairs}")
        return None

    if done_pairs.is_file():
        (comm["-23", str(src_pairs), str(done_pairs)] > str(dst_pairs))()
        src_pairs.rename(dst_orig)
    else:
        src_pairs.rename(dst_orig)
        dst_pairs.write_bytes(dst_orig.read_bytes())

    return dst_pairs


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
    
    dst = _eval_pairs(qdir, src, rdir)
    if not dst:
        return 1
    
    log.success(f"Ready for evalpairs: {dst}")
    return 0
