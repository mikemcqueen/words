def help_summary():
    return "Usage: wf wipe queue NUM"


def run(command, opts, argv):
    print(f"{command}: not yet implemented (dir={opts.dir})")
    return 0


def help(command, opts, argv):
    print(help_summary())
    return 0
