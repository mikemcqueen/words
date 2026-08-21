# eval.py
#
# Open a batch: move a queued artifact into <phase>/eval/<slug>/ and prepare it
# for whatever does the actual evaluating.
#
# Unlike submit and complete, the phases here are not the same command with
# different constants -- p2 splits its pairs and raises notes against them,
# and p1 does nothing of the kind. So the shared run() carries the shape both
# phases agree on and hands the difference to prepare().

import argparse
import subprocess

from pathlib import Path

from workflow import batch, command, context, fs, log, usage


CHUNK_SIZE = 400


class Eval(command.Action):
    def __init__(self, phase: str, summary: str, glob: str = "*",
                 source_noun: str = "pairs", ready_for: str = "evalpairs"):
        super().__init__(summary=summary, positional="SLUG")
        self.phase = phase
        self.glob = glob
        self.source_noun = source_noun
        self.ready_for = ready_for

    def parser(self):
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--no-filter", action="store_true",
                       help="skip filtering already-evaluated pairs")
        return p

    def prepare(self, pairs: Path, ctx) -> None:
        """What this phase does with its pairs once the batch is open."""

    def run(self, command, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command))

        # The positional is the batch directory name, and the queued artifact
        # is found under it by prefix.
        ctx = context.Context(root=opts.dir, phase=self.phase,
                              force=opts.force, slug=rest[0])
        pairs = batch.begin(ctx, glob=self.glob)
        log.info(f"{fs.line_count(pairs)} source {self.source_noun}")
        if not opts.no_filter:
            pairs = batch.filter_done(pairs, ctx)

        self.prepare(pairs, ctx)

        # TODO: (optionally?) copy file to somewhere specified by user

        log.success(f"{fs.line_count(pairs)} pairs ready for {self.ready_for}: "
                    f"{pairs.name}")
        return 0


def get_split_paths(prefix: str, n_files: int, suffix: str = '') -> list[Path]:
    assert n_files < 27, "got some work to do"
    return [Path(f"{prefix}.a{chr(ord('a') + i)}{suffix}") for i in range(n_files)]


def get_split_file_count(path: Path, chunk_size=CHUNK_SIZE) -> int:
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


class EvalYes(Eval):
    def __init__(self):
        super().__init__(phase="p2", summary="p2      — evaluate yes pairs",
                         glob="*.p1.yes", source_noun="YES pairs",
                         ready_for="manual filtering")

    def prepare(self, pairs: Path, ctx) -> None:
        _make_notes(_split_pairs(pairs, f"/tmp/{pairs.name}"))


class EvalNo(command.Action):
    def __init__(self):
        super().__init__(summary="p3      — evaluate no pairs")

    def show_help(self, command, opts, argv) -> int:
        return usage.default_help(self.summary, argv, "usage: wf eval no [options]")

    def run(self, command, opts, argv) -> int:
        # TODO: implement no pairs evaluation
        return 0


P1 = Eval("p1", "p1      — evaluate pairs")
P2 = EvalYes()
P3 = EvalNo()
