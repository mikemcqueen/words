# eval_yes.py

import argparse
import subprocess
from pathlib import Path
from workflow import batch, config, eval_pairs, fs, usage, log


CHUNK_SIZE = 400


def help_summary(name) -> str:
    return "p2      — evaluate yes pairs"


def _format_help(command, opts, argv) -> str:
    return usage.format_help(command, help_summary(command), _make_parser(), "SLUG")


def show_help(command, opts, argv) -> int:
    text = _format_help(command, opts, argv)
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0


def _make_parser():
    p = argparse.ArgumentParser(add_help=False)
    eval_pairs.add_no_filter_flag(p)
    return p


def _parse_args(argv, opts):
    local_opts, rest = _make_parser().parse_known_args(argv)
    vars(opts).update(vars(local_opts))
    return rest, opts


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


def _eval_yes(slug: str, opts) -> Path:
    dst_pairs = batch.begin(opts, "p2", slug, glob="*.p1.yes")
    log.info(f"{fs.line_count(dst_pairs)} source YES pairs")
    if not opts.no_filter:
        dst_pairs = eval_pairs.filter_done_pairs(dst_pairs, "p2", opts)
    split_prefix = f"/tmp/{dst_pairs.name}"
    split_paths = _split_pairs(dst_pairs, split_prefix)
    _make_notes(split_paths)
    return dst_pairs


def run(command, opts, argv) -> int:
    argv, opts = _parse_args(argv, opts)
    if not argv:
        return usage.missing_argument(_format_help(command, opts, argv))

    dst_pairs = _eval_yes(argv[0], opts)
    if not dst_pairs:
        return 1
    
    # TODO: (optionally?) copy file to somewhere specified by user

    log.success(f"{fs.line_count(dst_pairs)} pairs ready for manual filtering: {dst_pairs.name}")
    return 0
