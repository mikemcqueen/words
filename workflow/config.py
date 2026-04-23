# config.py

from dataclasses import dataclass
from pathlib import Path
from workflow import fs


_PHASE1 = {
    "description": "First-pass: automated YES/NO classification by evalpair.",
    # TODO: "alias": "phase1",
    "parts" : {
        "queued": {
            "description": ("Pairs files queued for processing by evalpair. Use "
                            "`wf submit pairs` to enqueue.")
        },
        "running": {
            "description": ("Pair files in active processing by evalpair, and result "
                            "files being actively updated.")
        },
        "done": {
            "description": ("Pairs files and their associated result files that have "
                            "completed automated classification by evalpair.")
        }
    }
}


_PHASE2 = {
    "description": "Second-pass: manual review of evalpair-classified YES results.",
    "parts": {
        "queued": {
            "description": "Evalpair-classified YES result files queued for manual review."
        },
        "reviewing": {
            "description": "Evalpair-classified YES result files being manually reviewed."
        },
        "done": {
            "description": "Not sure what goes here. Might be unnecessary."
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


@dataclass(frozen=True)
class LayoutArgs:
    _parts: tuple[str, ...]
    _node: dict
    _invalid: str | None = None

    @property
    def is_leaf(self) -> bool:
        return "parts" not in self._node

    # an invalid argument was encountered
    @property
    def has_invalid(self) -> bool:
        return self._invalid is not None

    # i don't undertand how this works. boolean magic.
    @property
    def has_missing(self) -> bool:
        return not self._parts or not self.is_leaf

    @property
    def ok(self) -> bool:
        return not (self.has_invalid or self.has_missing)


def layout_args(argv: list[str]) -> LayoutArgs:
    node = CONFIG_LAYOUT
    consumed: list[str] = []

    for name in argv:
        allowed = node.get("parts", {})
        if not allowed or name not in allowed:
            return LayoutArgs(_parts=tuple(consumed), _node=node, _invalid=name)
        #if name not in allowed:
        #return LayoutArgs(parts=tuple(consumed), node=node, invalid=name), False
        node = allowed[name]
        consumed.append(name)

    return LayoutArgs(_parts=tuple(consumed), _node=node)


def _root_parts() -> dict:
    return CONFIG_LAYOUT["parts"]


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
