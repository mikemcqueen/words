"""Extract top-token YES probabilities from eval-result JSONL files."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import TextIO

import numpy as np

from src import compare_native
from src.common import prefetch


def write_yes_probabilities(paths: list[Path], output: TextIO) -> int:
    """Write one probability per YES-leading direction and return the count."""
    count = 0
    for path in paths:
        blocks = compare_native.iter_projected_blocks([str(path)], chunk_size=8192)
        for block in prefetch(blocks):
            labels = np.asarray(block.labels())[0]
            probabilities = np.asarray(block.probs())[0]
            values = probabilities[labels == compare_native.LABEL_YES]
            if values.size:
                output.write("".join(f"{value:.2f}\n" for value in values))
                count += int(values.size)
    return count


def _result_paths(inputs: list[Path]) -> list[Path]:
    paths = []
    for path in inputs:
        if path.is_dir():
            matches = sorted(path.glob("*.jsonl"))
            if not matches:
                raise SystemExit(f"no .jsonl files in {path}")
            paths.extend(matches)
        else:
            paths.append(path)
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the probability from every direction whose top token is YES. "
            "Output is one 0-1 probability per line."
        )
    )
    parser.add_argument(
        "results", nargs="+", type=Path,
        help="result .jsonl file or directory of .jsonl files",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output file (default: stdout)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = _result_paths(args.results)

    if args.output is not None:
        output_path = args.output.resolve()
        if output_path in {path.resolve() for path in paths}:
            raise SystemExit(f"output path is also an input: {args.output}")

    if args.output is None:
        count = write_yes_probabilities(paths, sys.stdout)
    else:
        with args.output.open("w") as output:
            count = write_yes_probabilities(paths, output)

    print(
        f"wrote {count:,} YES-leading probabilities from {len(paths)} file(s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
