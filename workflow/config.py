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


_BEST = {
    "description": "BEST PAIRS workflow state",
    "parts": {
        "idx": {
            "description": "shared Nutrimatic indexes"
        }
    }
}


# The dictionary is root-global, not a BEST input. Four tools read the derived
# file in their workflow-configured modes -- dfs-anagrams under --dict, and the
# three *-segments tools under --wf/--wfroot -- and a future p1 filter will read
# it from outside BEST entirely, so it sits beside classified/ rather than under
# one consumer.
#
# "content": True is load-bearing and not decorative. `LayoutArgs.has_missing`
# is `not (parts and (is_leaf or has_content))`, so without it `wf show dict`
# would stop listing and demand a subpart -- and dict holds files (the base and
# the derived dictionary) as well as subparts. `done` carries it for the same
# reason: reviewed.words sits beside in/.
_DICT = {
    "description": "The shared Nutrimatic dictionary and its removals",
    "content": True,
    "parts": {
        "removed": {
            "description": "applied word-removal generations"
        },
        "done": {
            "description": "completed removal rounds",
            "content": True,
            "parts": {
                "in": {
                    "description": "per-round submitted word inputs"
                }
            }
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
        "classified": _CLASSIFIED,
        "dict":       _DICT,
        "best":       _BEST
    }
}


# The hand-placed base, never written by the workflow, and the derived file
# every workflow-configured consumer reads. Named here rather than beside any
# one of them: naming a shared artifact after its consumer is how a dictionary
# that meant two different things on two sides would start.
DICTIONARY_BASE_NAME = "words.big"
DICTIONARY_NAME = "words.filtered"
REVIEWED_WORDS_NAME = "reviewed.words"


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


def base_dictionary(root_dir: Path) -> Path:
    """The hand-placed base: .wf/dict/words.big.

    It may be a symlink to an operator-chosen source; the target is their
    choice and is not part of the workflow contract. Nothing here ever writes
    it.
    """
    return path(root_dir, ["dict"]) / DICTIONARY_BASE_NAME


def dictionary(root_dir: Path) -> Path:
    """The derived dictionary: the base minus every recorded removal.

    Returning the path does not assert the file is there. `wf gen dict` creates
    it and `wf init` deliberately does not, so its absence is a first-build
    state for the rebuild and a missing required input for everything else --
    which is a distinction each read boundary makes for itself.
    """
    return path(root_dir, ["dict"]) / DICTIONARY_NAME


def removals(root_dir: Path) -> Path:
    """Where one round's removals land: .wf/dict/removed/."""
    return path(root_dir, ["dict", "removed"])


def reviewed_inputs(root_dir: Path) -> Path:
    """Where one round's submitted word input lands: .wf/dict/done/in/."""
    return path(root_dir, ["dict", "done", "in"])


def reviewed_words(root_dir: Path) -> Path:
    """The union of every reviewed input: .wf/dict/done/reviewed.words.

    Beside the per-round archive it is derived from, the way p1's done-set sits
    beside the inputs that fed it. It answers "has this word been looked at",
    which no removal record can: a word judged good is kept, and without this
    it returns to the top of every candidate listing for ever.
    """
    return path(root_dir, ["dict", "done"]) / REVIEWED_WORDS_NAME


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
