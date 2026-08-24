# Plan: Declarative filename extensions via config._EXTENSIONS

## Context
Three specific hardcoded extension ops get replaced: `_make_pairs_filename` in `submit_pairs.py` and two `_replace_suffix` calls in `complete_yes.py`. `complete_pairs.py` is untouched (result files, not pairs). `_EXTENSIONS` already exists on disc; two values need correcting.

## Critical Files
- `workflow/config.py` — add helpers + `apply_ext()` + `path_apply_ext()`


---

## Implementation

### Internal helpers

```python
def _subst(template: str, key: str | None) -> str:
    if "#key" in template:
        assert key, f"non-empty key required for template: {template}"
    return template.replace("#key", key) if key is not None else template

def _resolve_ref(ref: str) -> str:
    s, f = ref[1:].rsplit("/", 1)
    return _EXTENSIONS[s][f]

def _resolve_expect(entry: dict) -> str | None:
    if "expect" not in entry:
        return None
    v = entry["expect"]
    return _resolve_ref(v) if v.startswith("#") else v

def _apply_extension(name: str, stage: str, key: str | None) -> str:
    entry = _EXTENSIONS[stage]
    expected = _resolve_expect(entry)       # never key-substituted
    if expected is not None:
        assert name.endswith(expected), f"{name!r} doesn't end with {expected!r}"
    if "append" in entry:
        return name + _subst(entry["append"], key)
    if "replace" in entry:
        return name[:-len(expected)] + _subst(entry["replace"], key)
    return name
```

### `config.apply_ext(src: Path, key: str | None = None) -> Path`

Walks `src`'s parent directories forward from CONFIG_ROOT, 
collects all parts from there forward, validates via `config.path()`.
Returns new Path in same directory:

```python
def apply_ext(src: Path, key: str | None = None) -> Path:
    parts = src.parts
    wf_idx = parts.index(CONFIG_ROOT)
    root_dir = Path(*parts[:wf_idx])
    dir_parts = list(parts[wf_idx + 1:-1])
    path(root_dir, dir_parts)
    stage = "/".join(dir_parts)
    return src.parent / _apply_extension(src.name, stage, key)
```

### `config.path_apply_ext(root_dir: Path, parts: list[str], name: str, key: str | None = None) -> Path`

Stage from `"/".join(parts)`. Returns result in the `parts`-specified directory (validated via `config.path()`):

```python

def path_apply_ext(root_dir: Path, parts: list[str], name: str,
                   key: str | None = None) -> Path:
    stage = _EXTENSIONS["/".join(parts)]
    return path(root_dir, parts) / _apply_extension(name, stage, key)
```

---

## Caller changes

**`submit_pairs.py`** — remove `_make_pairs_filename`; stage comes from parts ("p1/queued"):
```python
dst = config.path_apply_ext(opts.dir, ["p1", "queued"], src.name)
```
** `complete_pairs.py** 
```python
yes_results = config.path_apply_ext(opts.dir, ["p2", "queued"], src_pairs.name, "yes")
no_results = config.path(opts.dir, ["p3", "queued"]) / config.apply_ext(src_pairs, 'no")
```

**`complete_yes.py`** — replace both `_replace_suffix` calls; remove the function:
```python
yes_pairs = config.apply_ext(src_pairs, key="yes")
no_pairs = no_pairs_dir / config.apply_ext(src_pairs, "no").name
```

---

## Verification

```bash
source ../.torch/bin/activate
python -c "
from pathlib import Path
from workflow import config
p1e = Path('/x/.wf/p1/eval/foo.pairs')
print(config.apply_ext(p1e, key='yes'))   # /x/.wf/p1/eval/foo.pairs.p1.yes
p2e = Path('/x/.wf/p2/eval/foo.pairs.p1.yes')
print(config.apply_ext(p2e, key='yes'))   # /x/.wf/p2/eval/foo.pairs.p2.yes
print(config.apply_ext(p2e, key='no'))    # /x/.wf/p2/eval/foo.pairs.p2.no
"
python -m pytest tests/ -x -q
```
