from workflow import wipe_queue
from workflow.dispatch import dispatch_run, dispatch_help

SUBCOMMANDS = {"queue": wipe_queue}


def help_summary():
    return "wipe    — wipe workflow state (queue)"


def run(argv):
    return dispatch_run(SUBCOMMANDS, argv)


def help(argv):
    return dispatch_help(SUBCOMMANDS, argv)
