# config.py

from dataclasses import dataclass
from pathlib import Path
from workflow import fs, setops


_PHASE1 = {
    "description": "First-pass: automated YES/NO classification by evalpair.",
    # TODO: "alias": "phase1",
    "parts" : {
        "queued": {
            "description": ("Pair files queued for processing by evalpair. Use "
                            "`wf submit pairs` to enqueue.")
        },
        "eval": {
            "description": ("Pair files in active processing by evalpair, and result "
                            "files being actively updated.")
        },
        "done": {
            "description": ("Pair files and their associated result files that have "
                            "completed automated classification by evalpair."),
            "content": True,
            "parts": {
                "in": { "description": "p1 eval input archive (raw pairs)" },
                "out": { "description": "p1 eval output archive (evalpair results jsonl)" }
            }
        }
    }
}


_PHASE2 = {
    "description": "Second-pass: manual review of evalpair-classified YES results.",
    "parts": {
        "queued": {
            "description": ("Evalpair-classified YES pair files queued for manual "
                            "classification.")
        },
        "eval": {
            "description": "Evalpair-classified YES pair files being manually classified."
        },
        "done": {
            "description": ("Evailpair-classified YES pair files and their associated "
                            "enex files, that have completed manual classification."),
            "content": True,
            "parts": {
                "in": {
                    "description": "p2 eval input (evalpair-classified YES pairs)"
                },
                "out": {
                    "description": "p2 eval output (manually classified enex)",
                    "content": True,
                    "parts": {
                        "enex": {
                            "description": "p2 eval output parts in raw evernote format"
                        }
                    }
                }
            }
        }
    }
}


_PHASE3 = {
    "description": "Third-pass: probably a 2nd automated pass of evalpair-classified NO results.",
    "parts": {
        "queued": {
            "description": "Evalpair-classified NO result files queued for a 2nd automated pass."
        }
    }
}


# stable_mtime marks a node whose aggregate is dated by other things: a
# reader compares its mtime against a derived artifact to ask whether that
# artifact predates the current set of verdicts. Rewriting it with unchanged
# content would answer that question wrongly, so a no-op fold must leave it
# alone. The phase done-sets carry no such flag -- nothing dates them, and
# p1_done.pairs is large enough that the compare would not be free.
_CLASSIFIED = {
    "description": "Classified pairs (and their results?)",
    "parts": {
        "yes": {
            "description": "yes",
            "stable_mtime": True
        },
        "no": {
            "description": "no",
            "stable_mtime": True
        },
        "all": {
            "description": "all"
        }
    }
}


CONFIG_ROOT = ".wf"


CONFIG_LAYOUT = {
    "description": "workflow layout",
    "parts": {
        "p1":         _PHASE1,
        "p2":         _PHASE2,
        "p3":         _PHASE3,
        "classified": _CLASSIFIED
    }
}


@dataclass(frozen=True)
class LayoutArgs:
    parts: tuple[str, ...]
    node: dict
    _invalid: str | None = None

    @property
    def is_leaf(self) -> bool:
        return "parts" not in self.node

    @property
    def has_content(self) -> bool:
        return self.node.get("content", False)

    # an invalid argument was encountered
    @property
    def has_invalid(self) -> bool:
        return self._invalid is not None

    # boolean magic.
    @property
    def has_missing(self) -> bool:
        return not (self.parts and (self.is_leaf or self.has_content))

    @property
    def ok(self) -> bool:
        return not (self.has_invalid or self.has_missing)


def layout_args(argv: list[str]) -> LayoutArgs:
    node = CONFIG_LAYOUT
    consumed: list[str] = []

    for name in argv:
        allowed = node.get("parts", {})
        assert isinstance(allowed, dict)
        if not allowed or name not in allowed:
            return LayoutArgs(parts=tuple(consumed), node=node, _invalid=name)
        node = allowed[name]
        consumed.append(name)

    return LayoutArgs(parts=tuple(consumed), node=node)


def _root_parts() -> dict:
    parts = CONFIG_LAYOUT["parts"]
    assert isinstance(parts, dict)
    return parts


def path(root_dir: Path, parts: list[str]) -> Path:
    path = root_dir / CONFIG_ROOT
    fs.raise_if_not_dir(path)

    all_parts: list[str] = []
    allowed_parts: dict = _root_parts();
    for name in parts:
        if not name in allowed_parts:
            raise ValueError(f"{' '.join(all_parts)}/{name} is not part of the layout configuration")
        all_parts.append(name)
        allowed_parts = allowed_parts[name]["parts"] if "parts" in allowed_parts[name] else {}
        path = path / name
        fs.raise_if_not_dir(path)

    return path


def classified(root_dir: Path, kind: str) -> Path:
    """A global classified set: .wf/classified/<kind>/<kind>.pairs.

    Bundle-independent by construction. These are the workflow's standing
    verdicts about pairs, not a record of any one review batch, which is why
    they live beside the phases rather than inside one.
    """
    return path(root_dir, ["classified", kind]) / f"{kind}.pairs"


def stable_mtime(parts: list[str]) -> bool:
    """Whether this node's aggregate must keep its mtime across a no-op write."""
    return layout_args(parts).node.get("stable_mtime", False)


def fold_classified(root_dir: Path, kind: str, src: Path) -> Path:
    """Union src into the standing classified set for kind.

    The one way to write those aggregates. Which write policy they need is a
    property of the destination, not of the caller's errand, so it is looked up
    here from the layout rather than passed in: a caller that knows only which
    verdict it is recording cannot get it wrong, and cannot be left behind if
    the policy changes.
    """
    parts = ["classified", kind]
    return setops.fold(src, classified(root_dir, kind),
                       stable_mtime=stable_mtime(parts))
