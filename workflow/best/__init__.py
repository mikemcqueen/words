# A BEST PAIRS target is one sentence, minimum word length, and exact segment
# count: best/s2/m4/g4. The dynamic body is intentionally absent from
# CONFIG_LAYOUT; its shapes and state are derived from the files already on
# disk.

from workflow.best.commands import COMMAND


__all__ = ["COMMAND"]
