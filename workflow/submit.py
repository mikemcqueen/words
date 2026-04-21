from workflow import submit_pairs
from workflow.dispatch import dispatch_run, dispatch_help

SUBCOMMANDS = {"pairs": submit_pairs}


def help_summary(name):
    return "submit  — submit items (pairs)"


def run(command, opts, argv):
    return dispatch_run(command, SUBCOMMANDS, opts, argv)


def help(command, opts, argv):
    return dispatch_help(command, SUBCOMMANDS, opts, argv)
