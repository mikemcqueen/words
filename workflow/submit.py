# submit.py

from workflow import submit_pairs, usage, dispatch


SUBCOMMANDS = {"pairs": submit_pairs}


def help_summary(name):
    return "submit  — submit items (pairs)"


def run(command, opts, argv):
    return dispatch.run(command, SUBCOMMANDS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, SUBCOMMANDS, opts, argv)
