from workflow import complete_pairs
from workflow.dispatch import dispatch_run, dispatch_help

SUBCOMMANDS = {"pairs": complete_pairs}


def help_summary(name):
    return "complete— complete running jobs (pairs)"


def run(command, opts, argv):
    return dispatch_run(command, SUBCOMMANDS, opts, argv)


def help(command, opts, argv):
    return dispatch_help(command, SUBCOMMANDS, opts, argv)
