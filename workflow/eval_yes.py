# eval_yes.py

import subprocess
from pathlib import Path
from workflow import config, eval_pairs, fs, usage, log


CHUNK_SIZE = 750


def _usage_text():
    return "usage: wf eval yes PAIRS-FILE"


def help_summary(name) -> str:
    return "yes     — evaluate yes pairs"


def show_help(command, opts, argv) -> int:
    return usage.default_help(help_summary(command), argv, _usage_text())


def get_split_paths(prefix: str, n_files: int, suffix: str = '') -> list[Path]:
    assert n_files < 27, "got some work to do"
    return [Path(f"{prefix}.a{chr(ord('a') + i)}{suffix}") for i in range(n_files)]


def get_split_file_count(path: Path, chunk_size = CHUNK_SIZE) -> int:
    n_lines = fs.line_count(path)
    n_files = fs.line_count(path) // chunk_size
    return n_files + (1 if n_files * chunk_size < n_lines else 0)


def _split_pairs(pairs_file: Path, split_prefix: str) -> list[Path]:
    n_files = get_split_file_count(pairs_file)
    # check=True raises on non-zero return code
    subprocess.run(["split.sh", f"{pairs_file}", f"{CHUNK_SIZE}", f"{split_prefix}"],
                   stdout=subprocess.DEVNULL, check=True)
    paths = get_split_paths(split_prefix, n_files)
    fs.raise_if_any_not_file(paths)
    return paths
        

def _make_notes(paths: list[Path]) -> None:
    log.info(f"Creating {len(paths)} notes...")
    for path in paths:
        subprocess.run(["note", "-pf.72", "--create", f"{path}", "--text", "--checkbox",
                        "--production"], stdout=subprocess.DEVNULL, check=True)


def _eval_yes(src_pairs: Path, opts) -> Path:
    dst_pairs = eval_pairs.move_pairs_done(src_pairs, "p2", opts)
    split_prefix = f"/tmp/{dst_pairs.name}"
    split_paths = _split_pairs(dst_pairs, split_prefix)
    _make_notes(split_paths)

    return dst_pairs


def run(command, opts, argv) -> int:
    if not argv:
        details = _usage_text()
        return usage.missing_argument(details)

    src_dir = config.path(opts.dir, ["p2", "queued"]);
    src_pairs = src_dir / argv[0]
    fs.raise_if_not_file(src_pairs)

    dst_pairs = _eval_yes(src_pairs, opts)
    if not dst_pairs:
        return 1
    
    # TODO: (optionally?) copy file to somewhere specified by user

    log.success(f"Ready for round 2: {src_pairs.name}")
    return 0
