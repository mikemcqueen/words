# A BEST PAIRS target is one sentence, letter set, minimum word length, and
# exact segment count: best/s2/u-thisandthat/m4/g4. A full sentence has too
# many letters for dfs-anagrams and top-segments to sample usefully, so an
# entire run belongs to one subset of them, and the tree keys on it. The
# dynamic body is intentionally absent from CONFIG_LAYOUT; its shapes and
# state are derived from the files already on disk.

from workflow.best.commands import COMMAND


__all__ = ["COMMAND"]
