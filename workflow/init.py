# init.py

from pathlib import Path

from workflow import command, config, fs, log, usage


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


class Init(command.Action):
    def __init__(self):
        super().__init__(summary="init    — initialize a workflow (stub)")

    def run(self, command, opts, argv: list[str]) -> int:
        if argv:
            return usage.invalid_argument(argv[0],
                                          usage.default_help_text(self.summary))
        init(opts)
        log.success(f"Initialized {opts.dir}")
        return 0


COMMAND = Init()
