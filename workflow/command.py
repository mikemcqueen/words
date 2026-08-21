# command.py
#
# What a registry entry is.
#
# Every value in a registry -- COMMANDS in wf.py, or a Dispatcher's targets --
# answers three calls from dispatch.py:
#
#     help_summary(name)             -> str
#     run(command, opts, argv)       -> int
#     show_help(command, opts, argv) -> int
#
# Subclassing Command is not required to satisfy that. dispatch calls by name,
# so a plain module still participates unchanged -- which is why show.py, whose
# layout-driven help is a genuinely different shape, stays a module. The base
# class exists to hold the one definition Dispatcher and Action share, not to
# gatekeep.
#
# The split is between a command that hands off and a command that acts. Both
# used to be written as modules, and because a module cannot take arguments,
# every variant of one meant another near-identical file: submit_pairs.py and
# submit_yes.py differed in four tokens. An object takes arguments, so the
# variants become instances and the four tokens become a table.

from workflow import dispatch, usage


class Command:
    def __init__(self, summary: str = ""):
        self.summary = summary

    def help_summary(self, name) -> str:
        return self.summary

    def run(self, command, opts, argv) -> int:
        raise NotImplementedError

    def show_help(self, command, opts, argv) -> int:
        raise NotImplementedError


class Dispatcher(Command):
    """Hands off to named targets:  wf submit p1 ..."""

    def __init__(self, summary: str, targets: dict):
        super().__init__(summary)
        self.targets = targets

    def run(self, command, opts, argv) -> int:
        return dispatch.run(command, self.targets, opts, argv)

    def show_help(self, command, opts, argv) -> int:
        # The summary leads, then dispatch prints usage and the target list.
        if not argv:
            print(self.help_summary(command))
        return dispatch.show_help(command, self.targets, opts, argv)


class Action(Command):
    """Does the work itself:  wf submit p1 FILE"""

    def __init__(self, summary: str, positional: str | None = None):
        super().__init__(summary)
        self.positional = positional

    def parser(self):
        """An argparse parser for this command's own flags, or None."""
        return None

    def parse(self, opts, argv):
        """Fold local flags into opts; return the remaining positionals."""
        local = self.parser()
        if local is None:
            return argv
        local_opts, rest = local.parse_known_args(argv)
        vars(opts).update(vars(local_opts))
        return rest

    def format_help(self, command) -> str:
        return usage.format_help(command, self.summary, self.parser(),
                                 self.positional)

    def show_help(self, command, opts, argv) -> int:
        text = self.format_help(command)
        if argv:
            return usage.invalid_argument(argv[0], text)
        print(text, end="")
        return 0
