from workflow import show_all, show_queue
from workflow.dispatch import dispatch_run, dispatch_help

SUBCOMMANDS = {"all": show_all, "queue": show_queue}


def help_summary():
    return "show    — display workflow state (all | queue)"


def run(argv):
    return dispatch_run(SUBCOMMANDS, argv)


def help(argv):
    return dispatch_help(SUBCOMMANDS, argv)
