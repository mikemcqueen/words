# complete_yes.py

import subprocess
from pathlib import Path
from plumbum.cmd import cat, sort
from typing import Literal
from workflow import log, fs, config, usage, complete_pairs


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


MAX_NOTE_PARTS = 26


def _retrieve_notes(src_pairs: Path, enex_dir: Path, opts) -> list[Path]:
    # Download note parts .aa, .ab, .ac, ... until a part isn't found (max 26).
    # The part count is no longer derived from CHUNK_SIZE; we probe until the
    # note store reports "note not found".
    enex_paths: list[Path] = []
    for i in range(MAX_NOTE_PARTS):
        title = f"{src_pairs.name}.a{chr(ord('a') + i)}"
        enex_path = enex_dir / (title + ".enex")
        if not opts.force:
            fs.raise_if_exists(enex_path)
        result = subprocess.run(["note", "-pf.72", "--get", title, "--production"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            if "note not found" in result.stderr:
                break  # no more parts
            raise RuntimeError(f"failed to retrieve note {title}:\n{result.stderr}")
        enex_path.write_text(result.stdout)
        enex_paths.append(enex_path)
    if not enex_paths:
        raise RuntimeError(f"no note parts found for {src_pairs.name} (expected at least .aa)")
    return enex_paths
                           
                           
def _phase2_name(name: str, yesno: YesNo) -> str:
    # <base>[.p1[.<middle>]].yes -> <base>.p2[.<middle>].<yesno>
    assert name.endswith(".yes"), f"{name} doesn't end with .yes"
    stem = name[:-len(".yes")]
    base, sep, middle = stem.rpartition(".p1.")
    if sep:
        return f"{base}.p2.{middle}.{yesno}"
    if stem.endswith(".p1"):
        stem = stem[:-len(".p1")]
    return f"{stem}.p2.{yesno}"


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

    # Download note parts (.aa, .ab, ...) into .enex files, stopping at the
    # first part that doesn't exist (decoupled from CHUNK_SIZE).
    enex_dir = config.path(opts.dir, [phase, "done", "out", "enex"])
    enex_paths = _retrieve_notes(src_pairs, enex_dir, opts)
    log.info(f"downloaded {len(enex_paths)} notes")
    
    # Extract all YES pairs from .enex files to a single file in p2/eval and process
    yes_pairs = src_pairs.parent / _phase2_name(src_pairs.name, "yes")
    _extract_pairs_to(enex_paths, "yes", yes_pairs, opts)
    log.info(f"extracted {fs.line_count(yes_pairs)} unique YES pairs")

    _process_yes_pairs(yes_pairs, opts)

    # Extract all NO pairs from .enex files to a single file in p3/queued
    no_pairs_dir = config.path(opts.dir, ["p3", "queued"])
    no_pairs = no_pairs_dir / _phase2_name(src_pairs.name, "no")
    _extract_pairs_to(enex_paths, "no", no_pairs, opts)

    # Merge src_pairs with "phase-2 classification done" pairs
    complete_pairs.merge_with_done_pairs(phase, src_pairs, opts)
        
    # Move yes_pairs → p2/done/out, src_pairs → p2/done/in
    done_yes_pairs = complete_pairs.move_to_done(phase, yes_pairs, "out", opts)
    complete_pairs.move_to_done(phase, src_pairs, "in", opts)

    log.success(f"{fs.line_count(done_yes_pairs)} YES pairs, saved to: {done_yes_pairs}")
    return 0


def run(command, opts, argv) -> int:
    if not argv:
        return usage.missing_argument(_format_help(command, opts, argv))

    src_dir = config.path(opts.dir, ["p2", "eval"])
    src_pairs = src_dir / argv[0]
    fs.raise_if_not_file(src_pairs)

    return _complete(src_pairs, opts)
