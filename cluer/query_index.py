#!/usr/bin/env python3
"""Query the whole-word cluedata clue index.

    python cluer/query_index.py "new york"
    python cluer/query_index.py -j "new york"
    python cluer/query_index.py -f terms.txt
    python cluer/query_index.py -f terms.txt --forward
    python cluer/query_index.py -f terms.txt -r
    python cluer/query_index.py -f - < terms.txt
    python cluer/query_index.py -e "new york"
    python cluer/query_index.py -a PEAR

Space-separated QUERY words, or comma-separated words on each line of FILE,
can occur anywhere in a clue, in any order. -f prints each input query that
matches at least one clue; -r also prints its matching clues after the query.
With -j, QUERY must be two alphanumeric words separated by one space, or each
FILE line must be two such words separated by a comma. Only clues containing
either phrase order with a literal space match.
With -e, one or two words use the same separators, and the entire clue must
equal the word or either two-word order, ignoring case. -e and -j cannot be
combined.
--forward requires words in input order. Without -j or -e, other clue words
may appear between them. With -j or -e, only the input phrase order matches.
QUERY and -a print matches as "offset clue -> answers", as in find.py. --json
changes result lines to JSON, including the query and answer reference low
bits. Queries with no matches emit nothing. -a finds an exact answer and
prints its linked clues.
"""

import argparse
from pathlib import Path

from index import DEFAULT_DATA, DEFAULT_INDEX, query, query_answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", nargs="?", metavar="QUERY",
                        help="space-separated clue words to find (all required)")
    parser.add_argument("-f", "--file", metavar="FILE",
                        help="one comma-separated query per line, or - for stdin")
    parser.add_argument("-r", "--results", action="store_true",
                        help="with -f, print matching clues after each query")
    parser.add_argument("-j", "--adjacent", action="store_true",
                        help="require two words adjacent in either order")
    parser.add_argument("-e", "--exact", action="store_true",
                        help="match the entire clue to one or two words")
    parser.add_argument("--forward", action="store_true",
                        help="require clue words in input order")
    parser.add_argument("-a", "--answer", metavar="ANSWER",
                        help="find clues linked to this exact answer")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="cluedata file (default: cluer/data/cluedata)")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                        help="index directory (default: cluer/data/index)")
    parser.add_argument("--json", action="store_true",
                        help="format displayed clue results as JSON")
    args = parser.parse_args()
    if sum(value is not None for value in
           (args.query, args.file, args.answer)) != 1:
        parser.error("provide exactly one of QUERY, -f/--file, or -a/--answer")
    if args.results and args.file is None:
        parser.error("--results requires -f/--file")
    if args.exact and args.adjacent:
        parser.error("--exact cannot be used with -j/--adjacent")
    if args.adjacent and args.answer is not None:
        parser.error("two words required for --adjacent")
    if args.exact and args.answer is not None:
        parser.error("--exact cannot be used with -a/--answer")
    if args.forward and args.answer is not None:
        parser.error("--forward cannot be used with -a/--answer")
    try:
        if args.answer is not None:
            query_answer(args.data, args.index, args.answer,
                         json_output=args.json)
        else:
            query(args.data, args.index, args.file, args.query,
                  json_output=args.json, show_results=args.results,
                  adjacent=args.adjacent, exact=args.exact,
                  forward=args.forward)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
