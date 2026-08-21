# names.py
#
# Artifact names are *rendered*, never parsed.
#
# A filename is a content key: a flat rendering of the dimensions that
# distinguish what is in the file. It is not a history of the operations that
# produced it.
#
#     <slug>.<classifier>.<kind>
#
#     slug        <batch>.<slice>    batch-invariant; equals the eval dir name
#     batch       stem of the originating p1 result file
#     slice       <pmin*100>.<prange*100>, e.g. 90.10
#     classifier  p1 | p2 | p3       whose verdict the file represents
#     kind        pairs | yes | no | jsonl
#
# Invariant dimensions come first, so the slug is a true prefix of every
# filename in a batch and can be hoisted into a directory name. A phase
# transition never renames an artifact: p2 derives a *new* artifact from p1's,
# and p1's keeps its name forever. Nothing here takes a name apart -- where a
# slug is needed, the slug is the input and the file is found by prefix.

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


def slug(batch: str, pmin: float, prange: float) -> str:
    """The batch-invariant prefix shared by every artifact of one batch."""
    if not batch:
        raise ValueError("batch may not be empty")
    return f"{batch}{SEP}{slice_segment(pmin, prange)}"


def artifact(slug: str, classifier: str, kind: str) -> str:
    """Render one artifact name under a slug."""
    if not slug:
        raise ValueError("slug may not be empty")
    _check(classifier, CLASSIFIERS, "classifier")
    _check(kind, KINDS, "kind")
    return f"{slug}{SEP}{classifier}{SEP}{kind}"


def ensure_kind(name: str, kind: str) -> str:
    """Give an externally-supplied file its kind suffix.

    `submit` is the only caller: it is the boundary where an arbitrary outside
    file becomes a repo-managed artifact, and the only place a name arrives
    already spelled by someone else.
    """
    _check(kind, KINDS, "kind")
    suffix = f"{SEP}{kind}"
    return name if name.endswith(suffix) else f"{name}{suffix}"
