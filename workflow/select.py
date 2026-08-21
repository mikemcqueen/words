# select.py
#
# Resolve a selector against a layout slot. This is the one primitive that is
# genuinely new: it replaces the two hand-rolled resolvers that each command
# used to carry.
#
# Selectors name what to find, never how a filename is spelled -- no caller
# recovers a dimension by taking a name apart. Where a slug is needed, the slug
# is the input and the file is found by prefix.

from pathlib import Path

from workflow import config, fs


ALL = "all"


def select(root: Path, slot: list[str], selector: str, glob: str = "*") -> list[Path]:
    """Resolve selector against root/<slot>, returning at least one file.

        all         every file in the slot matching glob, sorted
        name:<n>    exactly that filename
        stem:<s>    the single file whose name starts with <s>
        /abs/path   bypasses the slot entirely

    A bare value carrying no prefix is read as name:<value>, which is the form
    `wf extract p1 yes FILE` already accepts.
    """
    if selector != ALL and Path(selector).is_absolute():
        path = Path(selector)
        fs.raise_if_not_file(path)
        return [path]

    directory = config.path(root, slot)
    fs.raise_if_not_dir(directory)

    if selector == ALL:
        paths = sorted(p for p in directory.glob(glob) if p.is_file())
        if not paths:
            raise ValueError(f"no {glob} files in {directory}")
        return paths

    kind, sep, value = selector.partition(":")
    if not sep:
        kind, value = "name", selector

    if kind == "name":
        path = directory / value
        fs.raise_if_not_file(path)
        return [path]

    if kind == "stem":
        matches = sorted(p for p in directory.glob(glob)
                         if p.is_file() and p.name.startswith(value))
        if not matches:
            raise ValueError(f"no file found for {value!r} in {directory}")
        if len(matches) > 1:
            names = ", ".join(p.name for p in matches)
            raise ValueError(f"multiple files found for {value!r} in {directory}: {names}")
        return matches

    raise ValueError(f"unknown selector: {selector!r}")
