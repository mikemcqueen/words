#!/usr/bin/env python3
"""Build the whole-word cluedata index.

    python cluer/build_index.py
    python cluer/build_index.py --replace

The destination must not exist unless --replace is given. Clue words use
ASCII boundaries, with apostrophes deleted, as described in cluer-index.md.
"""

import argparse
from pathlib import Path

from index import DEFAULT_DATA, DEFAULT_INDEX, build


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="cluedata file (default: cluer/data/cluedata)")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                        help="index directory (default: cluer/data/index)")
    parser.add_argument("--replace", action="store_true",
                        help="replace an existing index after a successful build")
    args = parser.parse_args()
    try:
        build(args.data, args.index, args.replace)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
