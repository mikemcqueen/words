import argparse
import sys
from pathlib import Path

from workflow import init, show, submit, wipe
from workflow.dispatch import dispatch_run

COMMANDS = {
    "init":    init,
    "show":    show,
    "submit":  submit,
    "wipe":    wipe
}


def _make_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-d", "--dir", type=Path)
    p.add_argument("-h", "--help", action="store_true")
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    opts, rest = _make_parser().parse_known_args(argv)
    if opts.dir is not None and not opts.dir.is_dir():
        print(f"-d/--dir: not a directory: {opts.dir}", file=sys.stderr)
        return 2
    opts.dir = (opts.dir or Path.cwd()).resolve()
    if opts.help and not (rest and rest[0].lower() == "help"):
        rest = ["help"] + rest
    return dispatch_run(None, COMMANDS, opts, rest)


if __name__ == "__main__":
    raise SystemExit(main())
