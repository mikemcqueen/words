from workflow import eval_pairs, usage, dispatch

SUBCOMMANDS = {"pairs": eval_pairs}


def help_summary(name):
    return "eval    — run evaluators (pairs)"


def run(command, opts, argv):
    return dispatch.run(command, SUBCOMMANDS, opts, argv)


def show_help(command, opts, argv):
    return dispatch.show_help(command, SUBCOMMANDS, opts, argv)
