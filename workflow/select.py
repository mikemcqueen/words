# select.py
#
# Resolve a selector against a layout slot. This is the one primitive that is
# genuinely new: it replaces the two hand-rolled resolvers that each command
# used to carry.
#
# Selectors name what to find, never how a filename is spelled -- no caller
# recovers a dimension by taking a name apart. Where a bundle name is needed,
# it is the input and the file is found by prefix.

from pathlib import Path

from workflow import config, fs


ALL = "all"


def select(root: Path, slot: list[str], selector: str, glob="*") -> list[Path]:
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
        paths = fs.globs(directory, glob)
        if not paths:
            raise ValueError(f"no {fs.spell(glob)} files in {directory}")
        return paths

    kind, sep, value = selector.partition(":")
    if not sep:
        kind, value = "name", selector

    if kind == "name":
        path = directory / value
        fs.raise_if_not_file(path)
        # The glob is the caller's shape contract, and naming a file exactly is
        # not a way around it: a stray `notes.txt` in the results slot has to
        # fail here, the way `stem:notes` already does, rather than reach a
        # reader that only knows how to say the JSON was malformed.
        if not fs.matches(path.name, glob):
            raise ValueError(
                f"{path.name} is not a {fs.spell(glob)} file in {directory}")
        return [path]

    if kind == "stem":
        matches = [p for p in fs.globs(directory, glob)
                   if p.name.startswith(value)]
        if not matches:
            # Naming the glob matters here: a file can be sitting in the slot
            # under a name the phase's queue contract does not admit, and
            # "no file found" alone points at the wrong thing.
            raise ValueError(f"no {fs.spell(glob)} file found for {value!r} "
                             f"in {directory}")
        if len(matches) > 1:
            # A prefix can be ambiguous by construction -- a phase whose queue
            # admits two shapes holds both under one bundle name -- so this
            # refuses to guess and says what to name instead.
            names = ", ".join(p.name for p in matches)
            raise ValueError(f"multiple files found for {value!r} in {directory}: "
                             f"{names} (name one of them exactly)")
        return matches

    raise ValueError(f"unknown selector: {selector!r}")
