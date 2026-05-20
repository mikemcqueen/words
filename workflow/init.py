# init.py

from pathlib import Path
from workflow import log, config, fs, usage


def help_summary(name):
    return "init    — initialize a workflow (stub)"


def ensure_dir(d: Path) -> Path:
    if not d.exists():
        d.mkdir()
    fs.raise_if_not_dir(d)
    return d


def ensure_layout(parent: Path, child: str, layout, opts) -> None:
    d = ensure_dir(parent / child)
    if "parts" in layout:
        for name in layout["parts"]:
            ensure_layout(d, name, layout["parts"][name], opts)


def init(opts) -> None:
    ensure_layout(opts.dir, config.CONFIG_ROOT, config.CONFIG_LAYOUT, opts)


def run(command: str, opts, argv: list[str]) -> int:
    if argv:
        return usage.invalid_argument(argv[0], usage.default_help_text(help_summary(command)))
    init(opts)
    log.success(f"Initialized {opts.dir}")
    return 0


def show_help(command: str, opts, argv: list[str]) -> int:
    text = usage.format_help(command, help_summary(command))
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0
