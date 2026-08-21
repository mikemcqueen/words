# wf_fixture.py
#
# Test fixtures for the workflow package: builds a throwaway .wf tree and the
# synthetic evalpair results that exercise src.filter's masking.
#
# The layout is always built by workflow.init, never hand-rolled, so a change to
# config.CONFIG_LAYOUT stays a one-line edit there.

import io
import json
import unittest

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src import compare_native
from workflow import config, init, usage, wf


# ---------------------------------------------------------------- workflow tree

def make_opts(root: Path, force: bool = False):
    """A global-opts namespace, built by the real parser so it stays in sync."""
    opts, _ = usage.make_global_parser().parse_known_args([])
    opts.dir = Path(root).resolve()
    opts.force = force
    return opts


def make_wf(root: Path, force: bool = False):
    """Initialize an empty .wf under root. Returns (opts, wf_dir)."""
    opts = make_opts(root, force)
    init.init(opts)
    return opts, opts.dir / config.CONFIG_ROOT


def slot(opts, parts: list[str]) -> Path:
    return config.path(opts.dir, parts)


def place(opts, parts: list[str], name: str, content: str = "") -> Path:
    """Write a file into a layout slot, e.g. place(opts, ["p1","queued"], "a.pairs")."""
    path = slot(opts, parts) / name
    path.write_text(content)
    return path


def make_batch(opts, phase: str, slug: str) -> Path:
    """Create an eval batch directory, as `eval` would."""
    directory = slot(opts, [phase, "eval"]) / slug
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_wf(*argv) -> tuple[int, str, str]:
    """Invoke the wf CLI in-process. Returns (exit_code, stdout, stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = wf.main(list(argv))
    return code, stdout.getvalue(), stderr.getvalue()


# ---------------------------------------------------------------- results jsonl

def _direction(tokens) -> list[dict]:
    if isinstance(tokens, tuple):
        tokens = [tokens]
    return [{token: prob} for token, prob in tokens]


def row(pair: str, seq: int, **directions) -> dict:
    """One evalpair result record.

    Each direction is (token, prob) or a list of them. Only the *first* entry is
    projected by the native reader -- it returns on the first token-prob field --
    so trailing entries exist to mirror real data, not to affect the mask.
    """
    return {
        "pair": pair,
        "seq": seq,
        "logprobs": {name: _direction(tokens) for name, tokens in directions.items()},
    }


def write_results(path: Path, rows: list[dict]) -> Path:
    with path.open("w") as f:
        for record in rows:
            json.dump(record, f)
            f.write("\n")
    return path


def write_pairs(path: Path, pairs: list[str]) -> Path:
    path.write_text("".join(f"{p}\n" for p in sorted(set(pairs))))
    return path


def pairs_of(rows: list[dict]) -> list[str]:
    return [r["pair"] for r in rows]


# ---------------------------------------------------------------- canonical rows

# Chosen to straddle the 0.9 band edge, to disagree across directions so
# _build_prob_mask's .any(axis=1) is actually exercised, and to diverge between
# .any() and max-over-directions at bands where pmin + prange != 1.0.
BAND_ROWS = [
    # squarely inside [0.9, 1.1)
    row("yes,high",      0, fwd=("YES", 0.95), rvs=("NO", 0.70)),
    # exactly on pmin -- included, the mask is >=
    row("yes,edge",      1, fwd=("YES", 0.90), rvs=("NO", 0.60)),
    # just under pmin -- excluded
    row("yes,below",     2, fwd=("YES", 0.8999), rvs=("NO", 0.62)),
    # p == 1.0, only inside because _pmax bumps the top to 1.1
    row("yes,one",       3, fwd=("YES", 1.0), rvs=("NO", 0.55)),
    # no YES in any direction -- the only kind of row the NO filter keeps
    row("no,both",       4, fwd=("NO", 0.99), rvs=("NO", 0.95)),
    # YES only in the reverse direction -- caught by .any(axis=1), not by fwd alone
    row("yes,rvsonly",   5, fwd=("NO", 0.70), rvs=("YES", 0.93)),
    # in-band YES against an out-of-band NO
    row("mixed,split",   6, fwd=("YES", 0.92), rvs=("NO", 0.88)),
    # .any() sees 0.60 in [0.5, 0.8); max-over-directions sees 0.95 and rejects.
    # Identical at 0.9/0.1, divergent at 0.5/0.3 -- this is the use_max witness.
    row("yes,divergent", 7, fwd=("YES", 0.60), rvs=("YES", 0.95)),
    # unrecognised token classifies as UNKNOWN: not YES, so the NO filter keeps it
    row("unknown,token", 8, fwd=("MAYBE", 0.99), rvs=("MAYBE", 0.97)),
]


# ---------------------------------------------------------------- native gating

def native_available() -> bool:
    """src._compare_native is a build artifact; `make build` produces it."""
    return compare_native._native_available()


requires_native = unittest.skipUnless(
    native_available(),
    "native extension not built (run: make build)")
