# submit.py

from workflow import submit_pairs, submit_yes, usage, dispatch


SUBCOMMANDS = {
    "p1": submit_pairs,
    "p2": submit_yes,
}


def help_summary(name):
    return "submit  — submit items (p1|p2)"


def run(command, opts, argv):
    return dispatch.run(command, SUBCOMMANDS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, SUBCOMMANDS, opts, argv)
