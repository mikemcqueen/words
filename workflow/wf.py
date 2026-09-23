import os
import subprocess
import sys
from pathlib import Path

from workflow import (
    best, classify, command, complete, dictionary, dispatch, extract,
    filter_pairs, history, init, log, notes, review, show, submit, usage,
    wipe, eval as evaluate,
)


COMMANDS = {
    "init":     init.COMMAND,
    "show":     show,
    "submit":   command.Dispatcher("submit  — submit items (p1|p2|words)",
                                   {"p1": submit.P1, "p2": submit.P2,
                                    "words": submit.WORDS}),
    "eval":     command.Dispatcher("eval     — run evaluators (p1|p2|p3|words)",
                                   {"p1": evaluate.P1, "p2": evaluate.P2,
                                    "p3": evaluate.P3,
                                    "words": evaluate.WORDS}),
    "notes":    command.Dispatcher("notes    — recreate evaluation notes (p2|words)",
                                   {"p2": notes.P2, "words": notes.WORDS}),
    "review":   command.Dispatcher("review   — submit and evaluate pairs (p1|p2)",
                                   {"p1": review.P1, "p2": review.P2}),
    "complete": command.Dispatcher("complete — complete evaluation (p1|p2|words)",
                                   {"p1": complete.P1, "p2": complete.P2,
                                    "words": complete.WORDS}),
    "extract":  command.Dispatcher("extract — extract archived results (p1)",
                                   {"p1": command.Dispatcher(
                                       "p1      — extract archived p1 results (yes)",
                                       {"yes": extract.P1_YES})}),
    "classify": command.Dispatcher("classify — record a standing verdict (yes|no)",
                                   {"yes": classify.YES, "no": classify.NO}),
    # Verb first, scope second, like every other root command: the scope names
    # the object, so words are removed and the dictionary is generated. Neither
    # takes a target -- dict/ is at the root and one removal applies to every
    # target. `wf best gen` stays where it is; that one generates a target's own
    # artifacts, and the dictionary stopped being one of those.
    "remove":   command.Dispatcher("remove   — record a removal verdict (words)",
                                   {"words": dictionary.REMOVE_WORDS}),
    "gen":      command.Dispatcher("gen      — generate a derived artifact (dict)",
                                   {"dict": dictionary.GEN_DICT}),
    "history":  history.COMMAND,
    "best":     best.COMMAND,
    # Unregistered until it is brought up to the steps architecture -- it is
    # the last pre-`steps/` command and now names its output differently from
    # `complete p1`. See the TODO at the top of filter_pairs.py.
#  ,"filter":  filter_pairs.COMMAND
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
    if opts.dir is not None:
        root, source = opts.dir, "-d/--dir"
    elif os.environ.get("WFROOT"):
        root, source = Path(os.environ["WFROOT"]), "$WFROOT"
    else:
        root, source = Path.cwd(), "current directory"
    if not root.is_dir():
        print(f"{source}: not a directory: {root}")
        return 2
    opts.dir = root.resolve()
    if opts.help:
        rest = ["help"] + rest
    return dispatch.run(None, COMMANDS, opts, rest)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        # The tool has diagnosed itself on stderr already; a traceback over
        # the top only buries it.
        #
        # cmd is a list at every call site in this package, but it is whatever
        # was handed to subprocess, and a string one -- shell=True, or a bare
        # string run -- would index to a single character and name a one-letter
        # command in the diagnostic.
        cmd = e.cmd
        program = cmd[0] if isinstance(cmd, (list, tuple)) else cmd
        log.error(f"{Path(program).name} failed ({e.returncode})")
        raise SystemExit(1)
    except (OSError, ValueError) as e:
        if isinstance(e, OSError) and e.strerror is not None:
            log.error(f"{e.strerror}: {e.filename}")
        else:
            log.error(str(e))
        raise SystemExit(1)
