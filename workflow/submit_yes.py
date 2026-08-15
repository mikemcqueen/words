from pathlib import Path

from plumbum.cmd import sort

from workflow import config, fs, log, usage


def help_summary(name):
    return "yes     — submit a YES pairs file into p2/queued (sorted, deduped)"


def _resolve_input(argv) -> Path:
    if not argv:
        raise ValueError("Missing FILE parameter.")

    src = Path(argv[0]).resolve()
    fs.raise_if_not_file(src)
    return src


def _make_yes_filename(fn: str) -> str:
    if not fn.endswith(".yes"):
        return fn + ".yes"
    return fn


def run(command, opts, argv):
    src = _resolve_input(argv)
    dst = config.path(opts.dir, ["p2", "queued"]) / _make_yes_filename(src.name)
    if not opts.force:
        fs.raise_if_exists(dst)

    (sort["-u", str(src)] > str(dst))()
    log.success(f"Submitted YES pairs {src.name}")
    return 0


def show_help(command, opts, argv):
    text = usage.format_help(command, help_summary(command), positional="YES-PAIRS-FILE")
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0
