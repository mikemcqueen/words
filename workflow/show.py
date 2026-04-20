from workflow import show_all, show_queue
from workflow.dispatch import dispatch_run, dispatch_help

SUBCOMMANDS = {"all": show_all, "queue": show_queue}


def help_summary():
    return "show    — display workflow state (all | queue)"


def run(command, opts, argv):
    return dispatch_run(command, SUBCOMMANDS, opts, argv)


def help(command, opts, argv):
    return dispatch_help(command, SUBCOMMANDS, opts, argv)
