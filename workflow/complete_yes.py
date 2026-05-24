# complete_yes.py

import subprocess
from pathlib import Path
from plumbum.cmd import cat, sort
from typing import Literal
from workflow import log, fs, config, usage, eval_yes, complete_pairs


YesNo = Literal["yes", "no"]


def help_summary(name):
    return "yes     — complete a yes file manual review"


def _format_help(command, opts, argv):
    return usage.format_help(command, help_summary(command), positional="PAIRS-FILE")


def show_help(command, opts, argv):
    text = _format_help(command, opts, argv)
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0


def _parse_note_files(paths: list[Path], yesno: YesNo) -> list[Path]:
    parsed: list[Path] = []
    for path in paths:
        out_path = Path(f"/tmp/{path.name}.parsed")
        with out_path.open("w") as f:
            subprocess.run(["note", "--parse-file", str(path), "--type", yesno,
                           "--lines"], stdout=f, check=True)
        parsed.append(out_path)
    return parsed


def _retrieve_notes(paths: list[Path], opts) -> list[Path]:
    enex_paths: list[Path] = []
    for path in paths:
        enex_path = path.parent / (path.name + ".enex")
        if not opts.force:
            fs.raise_if_exists(enex_path)
        with enex_path.open("w") as f:
            subprocess.run(["note", "-pf.72", "--get", path.name, "--production"],
                            stdout=f, check=True)
        enex_paths.append(enex_path)
    return enex_paths
                           
                           
def _replace_suffix(name: str, old_suffix: str, new_suffix: str) -> str:
    assert name.endswith(old_suffix), f"{name} doesn't end with {old_suffix}"
    return name[:-len(old_suffix)] + new_suffix


def _extract_pairs_to(enex_paths: list[Path], yesno: YesNo, dst_pairs: Path, opts):
    # Parse .enex files into separate pairs files
    pair_paths = _parse_note_files(enex_paths, yesno)
    if yesno == "yes":
        log.info(f"parsed {sum(fs.line_count(p) for p in pair_paths)} YES pairs")
    if not opts.force:
        fs.raise_if_exists(dst_pairs)
    # Concat all pairs files into a single file of all pairs
    ((cat[pair_paths] | sort["-u"]) > str(dst_pairs))()


def _process_yes_pairs(yes_pairs, opts) -> None:
    # Merge YES pairs with global "classified yes" pairs
    cls_yes_pairs = config.path(opts.dir, ["classified", "yes"]) / "yes.pairs"
    complete_pairs.merge_pairs(yes_pairs, cls_yes_pairs)
    # TODO: should classified/yes have an /in?


def _complete(src_pairs: Path, opts) -> int:
    log.info(f"found: {src_pairs.name}")
    phase = "p2"

    # Generate split_paths with filenames representing note names
    n_files = eval_yes.get_split_file_count(src_pairs)
    split_prefix = str(config.path(opts.dir, [phase, "done", "out", "enex"]) / src_pairs.name)
    split_paths = eval_yes.get_split_paths(split_prefix, n_files)

    # Download notes into .enex files
    enex_paths = _retrieve_notes(split_paths, opts)
    log.info(f"downloaded {len(enex_paths)} notes")
    
    # Extract all YES pairs from .enex files to a single file in p2/eval and process

    # TODO BORKEN - e.g. p1.90.100.yes - maybe we just need to replace the rfind("p1", "p2")
    #               or regex, so it works with p1.xx.yy.yes -> p2.xx.yy.no below
    yes_pairs = src_pairs.parent / _replace_suffix(src_pairs.name, ".p1.yes", ".p2.yes")
    _extract_pairs_to(enex_paths, "yes", yes_pairs, opts)
    log.info(f"extracted {fs.line_count(yes_pairs)} unique YES pairs")

    _process_yes_pairs(yes_pairs, opts)

    # Extract all NO pairs from .enex files to a single file in p3/queued

    # TODO BORKEN - e.g. p1.90.100.yes
    no_pairs_dir = config.path(opts.dir, ["p3", "queued"])
    no_pairs = no_pairs_dir / _replace_suffix(src_pairs.name, ".p1.yes", ".p2.no")
    _extract_pairs_to(enex_paths, "no", no_pairs, opts)

    # Merge src_pairs with "phase-2 classification done" pairs
    complete_pairs.merge_with_done_pairs(phase, src_pairs, opts)
        
    # Move src_pairs → p2/done/in, yes_pairs → p2/done/out
    _, done_yes_pairs = complete_pairs.move_to_done(phase, src_pairs, yes_pairs, opts)

    log.success(f"{fs.line_count(done_yes_pairs)} YES pairs, saved to: {done_yes_pairs}")
    return 0


def run(command, opts, argv) -> int:
    if not argv:
        return usage.missing_argument(_format_help(command, opts, argv))

    src_dir = config.path(opts.dir, ["p2", "eval"])
    src_pairs = src_dir / argv[0]
    fs.raise_if_not_file(src_pairs)

    return _complete(src_pairs, opts)
