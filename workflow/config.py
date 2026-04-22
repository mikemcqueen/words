import argparse

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


_CLASSIFIED = {
    "description": "classified pairs (and their results?)",
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


ROOT = ".wf"


LAYOUT = {
    "description": "workflow layout",
    "parts": {
        "p1":         _PHASE1,
        "p2":         _PHASE2,
        "classified": _CLASSIFIED
    }
}

# not strictly necessary. but abstracts out some of the ["parts"] checking
# from validate_parsed_args().
def _build_parse_tree(parts: any) -> any:
    tree = {}
    for name in parts:
        if "parts" in parts[name]:
            tree[name] = _build_parse_tree(parts[name]["parts"])
        else:
            tree[name] = None

    return tree


def validate_parsed_args(command: str, parser, args):
    command = ' '.join ([command, args.root])
    tree =  _build_parse_tree(LAYOUT["parts"])
    node = tree[args.root]
    #print(f"validate_path args.root {args.root}, args.path {args.path} parts {node.keys()}")
    consumed = []
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

    for name in LAYOUT["parts"]:
        p = sub.add_parser(name)
        p.add_argument("path", nargs="*")

    return parser


def path(parts: [str], opts) -> Path:
    path = opts.dir / config.ROOT
    fs.raise_if_not_dir(path)

    all_parts = []
    allowed_parts = LAYOUT["parts"]
    for part in parts:
        if not part in allowed_parts:
            raise ValueError(f"{' '.join(all_parts)}/{part} is not part of the layout configuration")
        all_parts.append(part)
        cur_parts = cur_parts[part]["parts"] if "parts" in cur_parts[part] else {}
        path = path / part
        fs.raise_if_not_dir(path)
    return path
