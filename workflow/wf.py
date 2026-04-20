import sys

from workflow import init, show, submit, wipe
from workflow.dispatch import dispatch_run

COMMANDS = {"init": init, "show": show, "submit": submit, "wipe": wipe}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    return dispatch_run(COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
