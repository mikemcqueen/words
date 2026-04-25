# complete_yes.py

import subprocess
from pathlib import Path
from plumbum.cmd import cat, sort
from typing import Literal
from workflow import log, fs, config, usage, eval_yes, complete_pairs


YesNo = Literal["yes", "no"]


def _usage_text():
    return "usage: wf complete yes PAIRS-FILE"


def help_summary(name):
    return "yes     — complete a yes file manual review"


def show_help(command, opts, argv):
    return print(_usage_text())


def _parse_note_files(paths: list[Path], yesno: YesNo) -> list[Path]:
    parsed: list[Path] = []
    for path in paths:
        out_path = Path(f"/tmp/{path.name}.parsed")
        with out_path.open("w") as f:
            subprocess.run(["note", "--parse-file", str(path), "--type", yesno,
                           "--lines"], stdout=f, check=True)
        parsed.append(out_path)
    return parsed


def _retrieve_notes(paths: list[Path]) -> list[Path]:
    enex_paths: list[Path] = []
    for path in paths:
        enex_path = path.parent / (path.name + ".enex")
        with enex_path.open("w") as f:
            subprocess.run(["note", "-pf.72", "--get", path.name, "--production"],
                            stdout=f, check=True)
        enex_paths.append(enex_path)
    return enex_paths
                           
                           
def _replace_suffix(path: Path, old_suffix: str, new_suffix: str) -> Path:
    name = path.name
    assert name.endswith(old_suffix), f"{name} doesn't end with {old_suffix}"
    return path.parent / (name[:-len(old_suffix)] + new_suffix)


def _extract_pairs(enex_paths: list[Path], yesno: YesNo, pairs: Path, opts):
    # Parse .enex files into separate pairs files
    pair_paths = _parse_note_files(enex_paths, "no")
    if not opts.force:
        fs.raise_if_exists(pair_paths)
    # Concat all pairs files into a single file of all pairs
    ((cat[pair_paths] | sort["-u"]) > str(pairs))()


def _process_yes_pairs(yes_pairs, opts) -> None:
    # Merge yes_pairs int global "classified yes" pairs
    cls_yes_pairs = config.path(opts.dir, ["classified", "yes"]) / "yes.pairs"
    complete_pairs.merge_pairs(yes_pairs, cls_yes_pairs)
    # TODO: should classified/yes have an /in?


def _process_no_pairs(yes_pairs, opts) -> None:
    return


def _complete(src_pairs: Path, opts) -> int:
    log.info(f"found: {src_pairs.name}")
    phase = "p2"

    # Generate split_paths with filenames representing note names
    n_files = eval_yes.get_split_file_count(src_pairs)
    split_prefix = str(config.path(opts.dir, [phase, "done", "out", "enex"]) / src_pairs.name)
    split_paths = eval_yes.get_split_paths(split_prefix, n_files)
    if not opts.force:
        fs.raise_if_any_exist(split_paths)

    # Download notes into .enex files
    enex_paths = _retrieve_notes(split_paths)
    
    # Extract yes pairs from .enex files → p2/eval and process
    yes_pairs = _replace_suffix(src_pairs, ".yes", ".p2.yes") #TODO: old = .p1.yes
    _extract_pairs(enex_paths, "yes", yes_pairs, opts)
    _process_yes_pairs(yes_pairs, opts)

    # Extract no pairs from .enex files → p3/queued
    no_pairs_dir = config.path(opts.dir, ["p3", "queued"])
    no_pairs = no_pairs_dir / _replace_suffix(src_pairs.name, ".yes", ".p2.no")
    _extract_pairs(enex_paths, "no", no_pairs, opts)

    # Merge src_pairs into "2nd-pass classification done" pairs
    complete_pairs.merge_with_done_pairs(phase, src_pairs, opts)
        
    # Move src_pairs → p2/done/in, yes_pairs → p2/done/out
    complete_pairs.move_to_done(phase, src_pairs, yes_pairs, opts)

    log.success(f"Completed yes pairs {src_pairs.name}")
    return 0


def run(command, opts, argv) -> int:
    if not argv:
        return usage.missing_argument(_usage_text())

    src_dir = config.path(opts.dir, ["p2", "eval"])
    src_pairs = src_dir / argv[0]
    fs.raise_if_not_file(src_pairs)

    return _complete(src_pairs, opts)
