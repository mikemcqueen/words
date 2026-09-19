#!/usr/bin/env python3
"""Look up crossword clues and their answers in cluedata.

The file begins with a little-endian 32-bit word count, followed by that many
one-byte-length-prefixed answer strings. A little-endian 32-bit clue count
then precedes the clue records. Each clue record contains a length-prefixed
clue string, a little-endian 32-bit answer count, and that many little-endian
32-bit references. The answer index is reference >> 1. The low bit is retained
in the file but its meaning is not yet established.

The later metadata region is outside this tool's scope.

By default, each whitespace-separated query word must occur somewhere in the
clue, case-insensitively. --exact matches the entire clue instead.
"""

import argparse
import mmap
import struct
from pathlib import Path


def read_u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def lookup(data_path, query, exact, limit, show_flags):
    with data_path.open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        offset = 4
        answers = []
        for _ in range(read_u32(data, 0)):
            length = data[offset]
            answers.append(data[offset + 1 : offset + 1 + length].decode("latin-1"))
            offset += 1 + length

        clue_count = read_u32(data, offset)
        offset += 4
        needle = query.encode("latin-1").lower()
        words = needle.split()
        matches = 0
        for _ in range(clue_count):
            record_offset = offset
            length = data[offset]
            clue = data[offset + 1 : offset + 1 + length]
            offset += 1 + length
            answer_count = read_u32(data, offset)
            offset += 4

            clue_lower = clue.lower()
            if (clue_lower == needle if exact else
                    all(word in clue_lower for word in words)):
                matches += 1
                if limit == 0 or matches <= limit:
                    references = struct.unpack_from(
                        f"<{answer_count}I", data, offset
                    )
                    decoded = [
                        answers[reference >> 1]
                        + (f" [bit={reference & 1}]" if show_flags else "")
                        for reference in references
                    ]
                    print(f"0x{record_offset:x} {clue.decode('latin-1')!r} -> "
                          f"{', '.join(decoded)}")
            offset += 4 * answer_count

        print(f"{matches} matching clues")
        if limit and matches > limit:
            print(f"Showing first {limit}; use --limit 0 to show all.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query", help="case-insensitive clue words to find (all required)"
    )
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "cluedata")
    parser.add_argument("--exact", action="store_true", help="match the entire clue")
    parser.add_argument("--limit", type=int, default=20, help="rows to show; 0 means all")
    parser.add_argument("--show-flags", action="store_true", help="show reference low bits")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    lookup(args.data, args.query, args.exact, args.limit, args.show_flags)


if __name__ == "__main__":
    main()
