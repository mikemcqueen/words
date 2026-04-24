import argparse
import sys
from pathlib import Path

from workflow import init, show, submit, wipe, eval as evaluate, complete, log, dispatch


COMMANDS = {
    "init":     init,
    "show":     show,
    "submit":   submit,
    "eval":     evaluate,
    "complete": complete
#  ,"wipe":    wipe
}


def _make_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-d", "--dir", type=Path)
    p.add_argument("-f", "--force", action="store_true")
    p.add_argument("-h", "--help", action="store_true")
    return p


def _normalize_help_argv(argv: list[str]) -> list[str]:
    saw_help = False
    normalized: list[str] = []
    for arg in argv:
        if arg.lower() == "help":
            saw_help = True
            continue
        normalized.append(arg)
    if saw_help:
        normalized.append("-h")
    return normalized


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    argv = _normalize_help_argv(argv)
    opts, rest = _make_parser().parse_known_args(argv)
    if opts.dir is not None and not opts.dir.is_dir():
        print(f"-d/--dir: not a directory: {opts.dir}")
        return 2
    opts.dir = (opts.dir or Path.cwd()).resolve()
    if opts.help:
        rest = ["help"] + rest
    return dispatch.run(None, COMMANDS, opts, rest)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as e:
        if isinstance(e, OSError):
            log.error(f"{e.strerror}: {e.filename}")
        else:
            log.error(str(e))
        raise SystemExit(1)
