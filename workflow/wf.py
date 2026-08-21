import sys
from pathlib import Path

from workflow import (
    command, complete, dispatch, extract, filter_pairs, init, log, show, submit,
    usage, wipe, eval as evaluate,
)


COMMANDS = {
    "init":     init.COMMAND,
    "show":     show,
    "submit":   command.Dispatcher("submit  — submit items (p1|p2)",
                                   {"p1": submit.P1, "p2": submit.P2}),
    "eval":     command.Dispatcher("eval     — run evaluators (p1|p2|p3)",
                                   {"p1": evaluate.P1, "p2": evaluate.P2,
                                    "p3": evaluate.P3}),
    "complete": command.Dispatcher("complete — complete evaluation (p1|p2)",
                                   {"p1": complete.P1, "p2": complete.P2}),
    "extract":  command.Dispatcher("extract — extract archived results (p1)",
                                   {"p1": command.Dispatcher(
                                       "p1      — extract archived p1 results (yes)",
                                       {"yes": extract.P1_YES})}),
    "filter":   filter_pairs.COMMAND,
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
