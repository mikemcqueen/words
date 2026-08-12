# extract_p1.py

from workflow import dispatch, extract_p1_yes


_TARGETS = {
    "yes": extract_p1_yes,
}


def help_summary(name):
    return "p1      — extract archived p1 results (yes)"


def run(command, opts, argv):
    return dispatch.run(command, _TARGETS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, _TARGETS, opts, argv)
