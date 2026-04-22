from pathlib import Path
from plumbum.cmd import comm

from workflow import log, fs, config

def help_summary(name):
    return "pairs   — move a queued pairs file to the running stage"


def help(command, opts, argv):
    print(help_summary(command))
    print("usage: wf eval pairs [pair-file]")
    return 0


def _resolve_pair_file(qdir: Path, argv) -> Path | None:
    if argv:
        name = argv[0]
        src_path = qdir / Path(name).name
        fs.raise_if_not_file(src_path)
        return src_path

    # TODO: see complete_pairs.py
    files = [p for p in qdir.iterdir() if p.is_file()]
    if not files:
        log.error("The pairs queue is empty.")
        return None
    if len(files) > 1:
        log.error("Missing FILE parameter. Use `wf show pairs` to see files.")
        return None
    return files[0]


def _eval_pairs(src_path: Path, dst_dir: Path, opts) -> Path:
    dst_path = dst_dir / src_path.name
    fs.raise_if_exists(dst_path)

    dst_orig_path = dst_dir / f"{src_path.name}.orig"
    fs.raise_if_exists(dst_orig_path)

    done_path = config.path(opts.dir, ["p1", "done", "pairs"])
    if done_path.is_file():
        (comm["-23", str(src_path), str(done_path)] > str(dst_path))()
        src_path.rename(dst_orig_path)
    else:
        src_path.rename(dst_orig_path)
        dst_path.write_bytes(dst_orig_path.read_bytes())

    return dst_path


def run(command, opts, argv):
    src_dir = config.path(opts.dir, ["p1", "queued"]);
    dst_dir = config.path(opts.dir, ["p1", "running"]);

    src_path = _resolve_pair_file(src_dir, argv)
    if src_path is None:
        return 2

    dst_path = _eval_pairs(src_path, dst_dir, opts)
    if not dst_path:
        return 1
    
    # TODO: (optionally?) copy file to somewhere specified by user

    log.success(f"Ready for evalpairs: {dst_path}")
    return 0
