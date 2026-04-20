from workflow import submit_pairs
from workflow.dispatch import dispatch_run, dispatch_help

SUBCOMMANDS = {"pairs": submit_pairs}


def help_summary():
    return "submit  — submit items (pairs)"


def run(argv):
    return dispatch_run(SUBCOMMANDS, argv)


def help(argv):
    return dispatch_help(SUBCOMMANDS, argv)
