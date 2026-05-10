# config.py

from dataclasses import dataclass
from pathlib import Path
from workflow import fs


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


_CLASSIFIED = {
    "description": "Classified pairs (and their results?)",
    "parts": {
        "yes": {
            "description": "yes"
        },
        "no": {
            "description": "no"
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


_EXTENSIONS = {
    "p1/queued": {
        "append": ".pairs"
    },
    "p1/eval": {
        "expect": "#p1/queued/append",
        "append": ".p1.#key"
    },
    "p2/eval": {
        "expect":  ".p1.yes",
        "replace": ".p2.#key"
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
