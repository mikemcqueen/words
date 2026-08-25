# names.py
#
# Artifact names are *rendered*, never parsed.
#
# A filename is a content key: a flat rendering of the dimensions that
# distinguish what is in the file. It is not a history of the operations that
# produced it.
#
#     <bundle_name>.<classifier>.<kind>
#
#     bundle_name  <result_stem>.<slice>  bundle-invariant; the eval dir name
#     result_stem  stem of the originating p1 result file
#     slice        <pmin*100>.<prange*100>, e.g. 90.10
#     classifier   p1 | p2 | p3           whose verdict the file represents
#     kind         pairs | yes | no | jsonl
#
# Invariant dimensions come first, so the bundle name is a true prefix of every
# filename in a bundle and can be hoisted into a directory name. A phase
# transition never renames an artifact: p2 derives a *new* artifact from p1's,
# and p1's keeps its name forever. Nothing here takes a name apart -- where a
# bundle name is needed, it is the input and the file is found by prefix.

SEP = "."

CLASSIFIERS = ("p1", "p2", "p3")
KINDS = ("pairs", "yes", "no", "jsonl")


def _check(value: str, allowed: tuple[str, ...], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"unknown {label}: {value!r} (expected one of {', '.join(allowed)})")
    return value


def slice_segment(pmin: float, prange: float) -> str:
    """The probability band, rendered delimiter-free: 0.9/0.1 -> '90.10'."""
    return f"{round(pmin * 100)}{SEP}{round(prange * 100)}"


def bundle_name(result_stem: str, pmin: float, prange: float) -> str:
    """The bundle-invariant prefix shared by every artifact of one bundle."""
    if not result_stem:
        raise ValueError("result stem may not be empty")
    return f"{result_stem}{SEP}{slice_segment(pmin, prange)}"


def artifact(bundle_name: str, classifier: str, kind: str) -> str:
    """Render one artifact name under a bundle name."""
    if not bundle_name:
        raise ValueError("bundle name may not be empty")
    _check(classifier, CLASSIFIERS, "classifier")
    _check(kind, KINDS, "kind")
    return f"{bundle_name}{SEP}{classifier}{SEP}{kind}"


# The canonical name shapes a phase's queue holds, most-generic first.
#
# p1 takes one shape. p2 takes two, because it has two producers: `complete p1`
# advances an already-classified `*.p1.yes`, while a candidate list assembled
# outside the workflow -- top DFS segments, say -- carries no p1 verdict and
# must not be spelled as though it did.
#
# One table, read by both ends. `submit` renders an outside name into the first
# shape unless the name already is one of them; `eval` finds the queued
# artifact by exactly these. Spelling the two independently is what previously
# let `submit p2` queue a `*.yes` that `eval p2` could never open.
QUEUE_SUFFIXES = {
    "p1": (f"{SEP}pairs",),
    "p2": (f"{SEP}pairs", f"{SEP}p1{SEP}yes"),
}


def _queue_suffixes(phase: str) -> tuple[str, ...]:
    if phase not in QUEUE_SUFFIXES:
        raise ValueError(f"phase has no queue contract: {phase!r}")
    return QUEUE_SUFFIXES[phase]


def queue_name(phase: str, name: str) -> str:
    """Give an externally-supplied file the name its queue slot expects.

    `submit` is the only caller: it is the boundary where an arbitrary outside
    file becomes a repo-managed artifact, and the only place a name arrives
    already spelled by someone else.

    A name that is already one of the phase's shapes is kept verbatim. That is
    what makes submission idempotent -- no `.pairs.pairs` on a resubmit -- and
    what keeps an advanced `*.p1.yes` from being restamped as an unclassified
    candidate list.
    """
    suffixes = _queue_suffixes(phase)
    if any(name.endswith(suffix) for suffix in suffixes):
        return name
    return f"{name}{suffixes[0]}"


def queue_stem(phase: str, name: str) -> str:
    """The bundle name under a queued artifact's name -- inverse of queue_name.

    `eval` is the only caller: its positional may name the queued file itself
    rather than the bundle, and the bundle name is that name with the queue
    suffix taken back off. This reads the same table `queue_name` writes with,
    so it undoes exactly what submission did. It is not a name being taken
    apart for a dimension -- no other segment is recovered, and a name the
    contract does not admit is rejected rather than guessed at.
    """
    # Longest first, so a contract whose shapes overlap (`.yes` and `.p1.yes`)
    # strips the specific one rather than whichever came first in the table.
    for suffix in sorted(_queue_suffixes(phase), key=len, reverse=True):
        if name.endswith(suffix):
            stem = name[:-len(suffix)]
            if not stem:
                raise ValueError(f"{name!r} is a bare queue suffix, not an artifact")
            return stem
    shapes = ", ".join(f"*{suffix}" for suffix in _queue_suffixes(phase))
    raise ValueError(f"{name!r} is not a {phase} queue shape (expected {shapes})")


def queue_globs(phase: str) -> tuple[str, ...]:
    """The globs that find a phase's queued artifact, wherever it has moved to."""
    return tuple(f"*{suffix}" for suffix in _queue_suffixes(phase))
