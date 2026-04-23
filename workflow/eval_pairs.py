from pathlib import Path
from plumbum.cmd import comm

from workflow import log, fs, config, usage

def help_summary(name):
    return "pairs   — move a queued pairs file to the running stage"


def show_help(command, opts, argv):
    return usage.default_help(help_summary(command), argv, "usage: wf eval pairs [pair-file]")


def _resolve_pair_file(qdir: Path, argv) -> Path | None:
    if argv:
        name = argv[0]
        src_path = qdir / Path(name).name
        fs.raise_if_not_file(src_path)
        return src_path

    # TODO: see complete_pairs.py
    files = [p for p in qdir.iterdir() if p.is_file()]
    if not files:
        log.warn("The pairs queue is empty.")
        return None
    if len(files) > 1:
        log.error("Missing FILE parameter. Use `wf show p1 queued` to see files.")
        return None
    return files[0]


def _eval_pairs(src_pairs: Path, opts) -> Path:
    dst_dir = config.path(opts.dir, ["p1", "running"]);
    dst_pairs = dst_dir / src_path.name
    fs.raise_if_exists(dst_path)

    dst_orig = dst_dir / f"{src_path.name}.orig"
    fs.raise_if_exists(dst_orig)

    done_pairs = config.path(opts.dir, ["p1", "done"]) / "pairs"
    if done_path.is_file():
        (comm["-23", str(src_pairs), str(done_pairs)] > str(dst_pairs))()
        src_pairs.rename(dst_orig)
    else:
        src_pairs.rename(dst_orig)
        dst_pairs.write_bytes(dst_orig.read_bytes())

    return dst_pairs


def run(command, opts, argv):
    src_dir = config.path(opts.dir, ["p1", "queued"]);
    src_pairs = _resolve_pair_file(src_dir, argv)
    if src_path is None:
        return 2

    dst_pairs = _eval_pairs(src_pairs, opts)
    if not dst_pairs:
        return 1
    
    # TODO: (optionally?) copy file to somewhere specified by user

    log.success(f"Ready for evalpairs: {dst_pairs}")
    return 0
