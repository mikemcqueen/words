# eval.py
#
# Open a bundle: move a queued artifact into <phase>/eval/<bundle_name>/ and
# prepare it
# for whatever does the actual evaluating.
#
# Unlike submit and complete, the phases here are not the same command with
# different constants -- p2 splits its pairs and raises notes against them,
# and p1 does nothing of the kind. So the shared run() carries the shape both
# phases agree on and hands the difference to prepare().

import argparse
import subprocess

from pathlib import Path

from workflow import bundle, command, config, context, fs, log, names, usage


CHUNK_SIZE = 400


def _resolve_queued(root: Path, phase: str, positional: str) -> tuple[str, str]:
    """Read eval's positional, which may name either a bundle or a queued file.

    Returns the bundle name -- the eval directory, and the prefix of every
    artifact derived in it -- along with the selector that finds the queued
    artifact to open.

    Exact match first. A positional that is the name of a file in the phase's
    queue is taken as that file, and the bundle name is what is left when the
    queue suffix comes off. Everything else is a bundle name, found by prefix,
    which stays the form to reach for.

    The exact form exists because a prefix can be ambiguous by construction:
    p2's queue admits two shapes, so `s6.90.10.pairs` submitted and
    `s6.90.10.p1.yes` advanced can share the slot and the prefix `s6.90.10`
    names both. Naming the file is how the user says which -- and it costs
    nothing, because the bundle name is still derived from the contract rather
    than from whatever the user typed.
    """
    # Only a bare filename can name something in the slot; a path with
    # separators in it is not a queued name and falls through to the prefix
    # form, which reports it as the miss it is.
    if Path(positional).name == positional:
        queued = config.path(root, [phase, "queued"]) / positional
        if queued.is_file():
            return names.queue_stem(phase, positional), f"name:{positional}"
    return positional, f"stem:{positional}"


class Eval(command.Action):
    def __init__(self, phase: str, summary: str,
                 source_noun: str = "pairs", ready_for: str = "evalpairs"):
        super().__init__(summary=summary, positional="BUNDLE-NAME|QUEUED-FILE")
        self.phase = phase
        self.source_noun = source_noun
        self.ready_for = ready_for

    def parser(self):
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--no-filter", action="store_true",
                       help="skip filtering already-evaluated pairs")
        return p

    def prepare(self, pairs: Path, ctx) -> None:
        """What this phase does with its pairs once the bundle is open."""

    def run(self, command, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command))

        bundle_name, selector = _resolve_queued(opts.dir, self.phase, rest[0])
        ctx = context.Context(root=opts.dir, phase=self.phase,
                              force=opts.force, bundle_name=bundle_name,
                              selector=selector)
        pairs = bundle.begin(ctx)
        log.info(f"{fs.line_count(pairs)} source {self.source_noun}")
        if not opts.no_filter:
            pairs = bundle.filter_done(pairs, ctx)

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
        # Both queue shapes are opened the same way. An advanced `*.p1.yes`
        # arrives carrying a p1 verdict and a submitted `*.pairs` does not, but
        # that difference is provenance, not procedure: manual review reads the
        # pairs either way.
        super().__init__(phase="p2", summary="p2      — evaluate pairs for manual review",
                         source_noun="pairs", ready_for="manual filtering")

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
