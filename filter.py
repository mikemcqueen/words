import argparse
import json

from common import parse_yesno_response


def iter_lines(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def all_no_none(obj):
    for dim_entries in obj["logprobs"].values():
        if not dim_entries:
            continue
        for tok, prob in dim_entries[0].items():
            if parse_yesno_response(tok) == "YES":
                return False
    return True


def any_yes(obj, args):
    for dim_entries in obj["logprobs"].values():
        if not dim_entries:
            continue
        for tok, prob in dim_entries[0].items():
            if parse_yesno_response(tok) == "YES" and args.prob_min <= prob < args.prob_max:
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    yesno = parser.add_mutually_exclusive_group(required=True)
    yesno.add_argument("-y", "--yes", action="store_true")
    yesno.add_argument("-n", "--no", action="store_true")
    parser.add_argument("--pm", "--prob-min", dest="prob_min", type=float, default=0.5)
    parser.add_argument("--pr", "--prob-range", dest="prob_range", type=float, default=1.0)
    args = parser.parse_args()

    args.prob_max = args.prob_min + args.prob_range
    if args.prob_max == 1.0:
        args.prob_max += 0.1

    for obj in iter_lines(args.file):
        keep = any_yes(obj, args) if args.yes else all_no_none(obj)
        if keep:
            print(obj["pair"])

if __name__ == "__main__":
    main()
