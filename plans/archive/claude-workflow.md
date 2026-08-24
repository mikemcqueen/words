# Plan: `wf` CLI scaffold

## Context
Create a new Python CLI tool at `workflow/wf.py` (directory currently empty) with commands `init`, `show`, `submit`, plus a `help` meta-command. Each real command lives in its own file (`init.py`, `show.py`, `submit.py`). The scaffold must cleanly accommodate future nesting (`wf COMMAND SUBCOMMAND ...`) where sub-commands also get their own files — added declaratively, without special-case dispatch code at each level.

Goal of this first iteration: lock in the dispatch shape so subsequent work is "drop a new module in, register it in one dict" — not "edit the router."

## Design approach (free-functions, no classes)

### 1. A single reusable dispatcher
One module — `workflow/dispatch.py` — provides two free functions used identically at every level of the tree:

```python
def dispatch_run(registry: dict[str, Any], argv: list[str]) -> int
def dispatch_help(registry: dict[str, Any], argv: list[str]) -> int
```

- `registry` is a `{name: module}` map. Modules are duck-typed (see §2).
- `dispatch_run(registry, argv)`:
  - If `argv` is empty or `argv[0] == "help"` → `dispatch_help(registry, argv[1:])`.
  - If `argv[0]` is in `registry` and the next token is `"help"` → call that module's `help(argv[2:])` (handles `wf COMMAND help [sub...]`).
  - Else if `argv[0]` is in `registry` → call that module's `run(argv[1:])`.
  - Else → unknown-command error, print top-level summary, return non-zero.
- `dispatch_help(registry, argv)`:
  - Empty `argv` → print each module's `SUMMARY` line (the `wf help` listing).
  - `argv[0]` in `registry` → call that module's `help(argv[1:])` (lets the command further delegate to its own sub-registry for nested help).
  - Else → unknown, list summaries, non-zero.

This is the only routing logic in the whole tool. It is used by `wf.py` *and* by any command module that later grows sub-commands.

### 2. Module contract (duck-typed)
Each command/sub-command module exposes three names as free functions/constants:

```python
SUMMARY: str                         # one-line syntax+description for `wf help`
def run(argv: list[str]) -> int      # execute
def help(argv: list[str]) -> int     # print help; may re-dispatch into own SUBCOMMANDS
```

No base class, no registration decorator — just these three names. A module that wants sub-commands additionally defines:

```python
SUBCOMMANDS: dict[str, ModuleType] = {"foo": foo, "bar": bar}
```

…and implements its own `run` / `help` as one-liners:

```python
def run(argv):  return dispatch_run(SUBCOMMANDS, argv)
def help(argv): return dispatch_help(SUBCOMMANDS, argv)
```

Same shape at every level → no special-case code as the tree grows.

### 3. `wf.py` entry point
Tiny. Just the top-level registry + `main`:

```python
from workflow import init, show, submit
from workflow.dispatch import dispatch_run

COMMANDS = {"init": init, "show": show, "submit": submit}

def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    return dispatch_run(COMMANDS, argv)

if __name__ == "__main__":
    raise SystemExit(main())
```

Adding a new top-level command = one import + one dict entry.

### 4. Initial command stubs
`init.py`, `show.py`, `submit.py` each start as leaf modules:

```python
SUMMARY = "init    — initialize a workflow (stub)"

def run(argv):  print("init: not yet implemented"); return 0
def help(argv): print(SUMMARY); return 0
```

No `SUBCOMMANDS` yet — they'd be added later by defining the dict and swapping `run`/`help` to the one-liners in §2.

### 5. Argv convention
- Every `run` / `help` receives the argv slice *after* its own name.
- No top-level argparse; each leaf parses its own args however it wants. Keeps the scaffold free of premature structure and lets each command pick its own flag style.

## Files to create
- `workflow/__init__.py` (empty; makes it a package so imports work)
- `workflow/dispatch.py` — `dispatch_run`, `dispatch_help` (the only shared logic)
- `workflow/wf.py` — top-level entry point + `COMMANDS` registry
- `workflow/init.py`, `workflow/show.py`, `workflow/submit.py` — stubs exposing `SUMMARY`, `run`, `help`

## Verification
- `python -m workflow.wf` → top-level summary listing (same as `wf help`).
- `python -m workflow.wf help` → same listing.
- `python -m workflow.wf help show` and `python -m workflow.wf show help` → both invoke `show.help([])` and print identically.
- `python -m workflow.wf show` → runs `show.run([])` stub.
- `python -m workflow.wf bogus` → unknown-command error, non-zero exit, summary printed.
- `python -m workflow.wf help bogus` → unknown, non-zero, summary printed.
- Smoke-extend: temporarily give `show.py` a trivial `SUBCOMMANDS = {"x": ...}` and confirm `wf show help x` and `wf show x` both route correctly with no edits to `dispatch.py` or `wf.py` — proves the declarative-expansion goal.

## Resolved decisions
- **Package layout**: `workflow/` is a Python package; run as `python -m workflow.wf`. Adds `workflow/__init__.py`; siblings imported cleanly as `from workflow import init`.
- **Bare `wf`**: equivalent to `wf help` — prints top-level summary listing, exits 0.
