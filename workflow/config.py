import argparse

from pathlib import Path

from . import fs


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

# not strictly necessary. but abstracts out some of the ["parts"] checking
# from validate_parsed_args().
def _build_parse_tree(layout: dict) -> dict:
    tree: dict[str, dict|None] = dict()
    for key in layout:
        if "parts" in layout[key]:
            tree[key] = _build_parse_tree(layout[key]["parts"])
        else:
            tree[key] = None

    return tree


def validate_parsed_args(command: str, parser, args):
    command = ' '.join ([command, args.root])
    tree =  _build_parse_tree(CONFIG_LAYOUT["parts"])
    node = tree[args.root]
    #print(f"validate_path args.root {args.root}, args.path {args.path} parts {node.keys()}")
    consumed: list[str] = []
    for name in args.path:
        if node is None:
            parser.error(f"{command} {' '.join(consumed)} does not take further arguments ({name})")

        if name not in node:
            parser.error(
                f"invalid choice {name!r} after {command} {' '.join(consumed)}; "
                f"choose from {', '.join(node.keys())}"
            )

        consumed.append(name)
        node = node[name]

    if isinstance(node, dict):
        parser.error(
            f"incomplete path {command} {' '.join(consumed)}; "
            f"expected one of: {', '.join(node.keys())}"
        )


def arg_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="root", required=True)

    for name in CONFIG_LAYOUT["parts"]:
        p = sub.add_parser(name)
        p.add_argument("path", nargs="*")

    return parser


def path(root_dir: Path, parts: list[str]) -> Path:
    path = root_dir / CONFIG_ROOT
    fs.raise_if_not_dir(path)

    all_parts: list[str] = []
    allowed_parts: dict = CONFIG_LAYOUT["parts"]
    for name in parts:
        if not name in allowed_parts:
            raise ValueError(f"{' '.join(all_parts)}/{name} is not part of the layout configuration")
        all_parts.append(name)
        allowed_parts = allowed_parts[name]["parts"] if "parts" in allowed_parts[name] else {}
        path = path / name
        fs.raise_if_not_dir(path)

    return path
