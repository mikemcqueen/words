from workflow import usage


def help_summary(name):
    return "p3      — evaluate no pairs"


def show_help(command, opts, argv):
    return usage.default_help(help_summary(command), argv, f"usage: wf eval no [options]")


def run(command, opts, argv):
    # TODO: implement no pairs evaluation
    return 0
