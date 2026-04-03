#!/usr/bin/env python3
"""Display false positives and false negatives from a yesno results file."""

import json
import sys

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    fp = []  # said YES but should be NO
    fn = []  # said NO but should be YES

    for result in data.get("results", []):
        if "\u2717" not in result:
            continue
        pair, verdict = result.rsplit(":", 1)
        verdict = verdict.strip()
        if verdict.startswith("YES"):
            fp.append(pair.strip())
        elif verdict.startswith("NO"):
            fn.append(pair.strip())

    print(f"FP ({len(fp)}):")
    print("-------")
    for pair in fp:
        print(f"{pair}")
    print(f"\nFN ({len(fn)}):")
    print("-------")
    for pair in fn:
        print(f"{pair}")

if __name__ == "__main__":
    main()
