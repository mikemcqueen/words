# extract.py

from workflow import dispatch, extract_p1


_TARGETS = {
    "p1": extract_p1,
}


def help_summary(name):
    return "extract — extract archived results (p1)"


def run(command, opts, argv):
    return dispatch.run(command, _TARGETS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, _TARGETS, opts, argv)
