# init.py

from pathlib import Path
from workflow import log, config, fs


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
    init(opts)
    log.success(f"Initialized {opts.dir}")
    return 0


def help(command: str, opts, argv: list[str]) -> int:
    print(help_summary(command))
    return 0
