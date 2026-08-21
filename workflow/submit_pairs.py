# submit_pairs.py

from pathlib import Path

from workflow import fs, log, config, names, setops, usage


def help_summary(name):
    return "p1      — submit a pairs file into p1/queued (sorted, deduped)"


def _resolve_input(argv) -> Path:
    if not argv:
        raise ValueError("Missing FILE parameter.")

    src = Path(argv[0]).resolve()
    fs.raise_if_not_file(src)
    return src


def run(command, opts, argv):
    src = _resolve_input(argv)
    dst = config.path(opts.dir, ["p1", "queued"]) / names.ensure_kind(src.name, "pairs")
    if not opts.force:
        fs.raise_if_exists(dst)

    setops.merge([src], dst)
    log.success(f"Submitted pairs {src.name}")
    return 0


def show_help(command, opts, argv):
    text = usage.format_help(command, help_summary(command), positional="PAIRS-FILE")
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0
