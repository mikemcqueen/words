# complete.py

from workflow import complete_pairs, complete_yes, usage, dispatch


SUBCOMMANDS = {"pairs": complete_pairs, "yes": complete_yes}


def help_summary(name):
    return "complete— complete running jobs (pairs)"


def run(command, opts, argv):
    return dispatch.run(command, SUBCOMMANDS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, SUBCOMMANDS, opts, argv)
