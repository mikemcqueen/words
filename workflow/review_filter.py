# review_filter.py
#
# The standing sets a candidate review must not ask about again.

import tempfile

from pathlib import Path

from workflow import config, fs, setops


def filter_review_pairs(src: Path, root: Path, dst: Path, *,
                        local_no_pairs: Path | None = None,
                        completed_pairs: Path | None = None,
                        pcomm: bool = False) -> Path:
    """Write the review candidates in ``src`` that still need a verdict.

    YES and NO are workflow-global standing verdicts.  A BEST target may add a
    local NO set whose meaning is confined to that target, while a phase may
    add its accumulated done-set to preserve its own already-reviewed filter.
    All exclusions are merged first because a target-local file is hand-edited
    and ``comm -23`` requires both inputs to be sorted and unique.

    ``pcomm`` swaps ``comm -23`` for ``pcomm -23``, which treats ``a,b`` and
    ``b,a`` as the same pair.
    """
    exclusions = [config.classified(root, kind) for kind in ("yes", "no")]
    if local_no_pairs is not None:
        exclusions.append(local_no_pairs)
    if completed_pairs is not None:
        exclusions.append(completed_pairs)
    for path in exclusions:
        fs.raise_if_not_file(path)

    with tempfile.TemporaryDirectory(prefix="wf-review-filter-") as tmp:
        excluded = setops.merge(exclusions, Path(tmp) / "excluded.pairs")
        if pcomm:
            return setops.pair_diff(src, excluded, dst)
        return setops.diff(src, excluded, dst)
