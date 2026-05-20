# eval.py

from workflow import eval_pairs, eval_yes, eval_no, usage, dispatch


SUBCOMMANDS = {
    "pairs": eval_pairs,
    "yes": eval_yes,
    "no": eval_no
}


def help_summary(name):
    return "eval    — run evaluators pairs|yes|no"


def run(command, opts, argv):
    return dispatch.run(command, SUBCOMMANDS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, SUBCOMMANDS, opts, argv)
