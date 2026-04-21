from pathlib import Path
from workflow import log, config

def help_summary(name):
    return "init    — initialize a workflow (stub)"


def ensure_dir(parent: Path, child: str, opts: any) -> Path|None:
    d = parent / child
    if not d.exists():
        d.mkdir()
    if not d.is_dir():
        log.warn(f"{d} exists and is not a directory")
        return None
    return d


def ensure_layout(parent: Path, child: str, layout: any, opts: any) -> Path|None:
    p_dir = ensure_dir(parent, child, opts)
    if not p_dir:
        return None
    
    all_ok = True
    if "parts" in layout:
        for name in layout["parts"]:
            d = ensure_layout(p_dir, name, layout["parts"][name], opts)
            all_ok &= (d is not None)
        
    return p_dir if all_ok else None


def init(opts: any) -> bool:
    return ensure_layout(opts.dir, config.ROOT, config.LAYOUT, opts)


def run(command: str, opts: any, argv: [str]) -> int:
    if not init(opts):
        log.error(f"Initialization failed for {opts.dir}")
        return 1
    
    log.success(f"Initialized {opts.dir}")
    return 0


def help(command: str, opts: any, argv: [str]) -> int:
    print(help_summary())
    return 0
