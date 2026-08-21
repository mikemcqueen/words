import sys
from pathlib import Path

from workflow import (
    command, complete, dispatch, extract, filter_pairs, init, log, show, submit,
    usage, wipe, eval as evaluate,
)


COMMANDS = {
    "init":     init,
    "show":     show,
    "submit":   command.Dispatcher("submit  — submit items (p1|p2)",
                                   {"p1": submit.P1, "p2": submit.P2}),
    "eval":     evaluate,
    "complete": complete,
    "extract":  extract,
    "filter":   filter_pairs,
#  ,"wipe":    wipe
}


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
    opts, rest = usage.make_global_parser().parse_known_args(argv)
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
        if isinstance(e, OSError) and e.strerror is not None:
            log.error(f"{e.strerror}: {e.filename}")
        else:
            log.error(str(e))
        raise SystemExit(1)
