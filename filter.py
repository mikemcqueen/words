import argparse
import json

from common import parse_yesno_response


def iter_lines(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def any_yes(obj):
    for dim_entries in obj["logprobs"].values():
        if dim_entries and parse_yesno_response(next(iter(dim_entries[0]))) == "YES":
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-y", "--yes", action="store_true")
    group.add_argument("-n", "--no", action="store_true")
    args = parser.parse_args()

    for obj in iter_lines(args.file):
        keep = any_yes(obj) if args.yes else not any_yes(obj)
        if keep:
            print(obj["pair"])


if __name__ == "__main__":
    main()
