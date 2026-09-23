"""Artifact-backed history of the latest observable workflow milestones."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from workflow import (
    bundle, command, config, dictionary, fs, generation, names, notes, usage,
)
from workflow.best import state
from workflow.context import Context


DEFAULT_COUNT = 5


@dataclass(frozen=True)
class Event:
    domain: str
    when_ns: int
    title: str
    details: tuple[str, ...] = ()
    source_name: str | None = None


def _time(when_ns: int) -> str:
    return datetime.fromtimestamp(when_ns // 1_000_000_000).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z")


def _newest(events: list[Event]) -> Event | None:
    return max(events, key=lambda event: (event.when_ns, event.title),
               default=None)


def _matching_files(directory: Path, patterns) -> list[Path]:
    """fs.globs, with dangling matching symlinks diagnosed rather than lost."""
    if isinstance(patterns, str):
        patterns = (patterns,)
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_symlink() and not path.exists():
                fs.raise_if_not_file(path)
    return fs.globs(directory, patterns)


def _exists(path: Path) -> bool:
    """exists(), except a dangling symlink is a broken scanned artifact."""
    if path.is_symlink() and not path.exists():
        fs.raise_if_not_file(path)
    return path.exists()


def _source(ctx: Context) -> Path | None:
    """An open bundle's source, or None for malformed/incomplete scratch."""
    try:
        globs = bundle.source_globs(ctx)
        _matching_files(ctx.bundle_dir, globs)
        return bundle.source(ctx)
    except ValueError:
        return None


def _best_reviews(root: Path) -> tuple[dict[Path, tuple[state.Target,
                                                         state.ReviewRound]],
                                       list[state.Target]]:
    """Recognized review locations, also enforcing BEST's tree invariants."""
    recognized: dict[Path, tuple[state.Target, state.ReviewRound]] = {}
    targets = state.targets(root)
    for target in targets:
        queued, evaluating, archived = state.review_locations(target)
        for round_ in (*queued, *evaluating, *archived):
            recognized[round_.path] = target, round_
    return recognized, targets


def _review_detail(recognized, path: Path) -> str | None:
    found = recognized.get(path)
    if found is None:
        return None
    target, round_ = found
    return (f"BEST {round_.kind} review: {target.address}, "
            f"round {round_.ordinal}")


def _p2_queue(root: Path, recognized) -> Event | None:
    candidates: list[Event] = []
    patterns = names.queue_globs("p2")
    for parts in (["p2", "queued"], ["p2", "done", "in"]):
        directory = config.path(root, parts)
        for path in _matching_files(directory, patterns):
            details = [f"source: {fs.line_count(path)} pairs"]
            review = _review_detail(recognized, path)
            if review is not None:
                details.append(review)
            candidates.append(Event(
                "p2.queue", path.stat().st_mtime_ns,
                f"P2 queued: {path.name}", tuple(details), path.name))

    evaluating = config.path(root, ["p2", "eval"])
    for bundle_dir in sorted(path for path in evaluating.iterdir()
                             if path.is_dir()):
        ctx = Context(root=root, phase="p2", bundle_name=bundle_dir.name)
        source = _source(ctx)
        if source is None:
            continue
        details = [f"source: {fs.line_count(source)} pairs"]
        review = _review_detail(recognized, bundle_dir)
        if review is not None:
            details.append(review)
        candidates.append(Event(
            "p2.queue", source.stat().st_mtime_ns,
            f"P2 queued: {source.name}", tuple(details), source.name))
    return _newest(candidates)


def _opening_time(bundle_dir: Path, source: Path) -> tuple[int, Path] | None:
    filtered = bundle.filtered(source)
    if filtered.is_symlink() and not filtered.exists():
        fs.raise_if_not_file(filtered)
    if filtered.exists():
        fs.raise_if_not_file(filtered)
        return filtered.stat().st_mtime_ns, filtered
    # The sentence file is written beside an unfiltered source too.
    held = [p for p in bundle_dir.iterdir()
            if p.name != f"{bundle_dir.name}.sentence"]
    if held == [source]:
        return bundle_dir.stat().st_mtime_ns, source
    return None


def _p2_open(root: Path, recognized) -> Event | None:
    candidates: list[Event] = []
    evaluating = config.path(root, ["p2", "eval"])
    for bundle_dir in sorted(path for path in evaluating.iterdir()
                             if path.is_dir()):
        ctx = Context(root=root, phase="p2", bundle_name=bundle_dir.name)
        source = _source(ctx)
        if source is None:
            continue
        opened = _opening_time(bundle_dir, source)
        if opened is None:
            continue
        when_ns, evaluated = opened
        details = [
            f"submitted: {fs.line_count(source)}, "
            f"filtered: {fs.line_count(evaluated)}, "
            f"notes: {notes.part_count(evaluated)}",
        ]
        sentence = bundle.sentence(ctx)
        if sentence is not None:
            details.append(f"sentence: {sentence}")
        review = _review_detail(recognized, bundle_dir)
        if review is not None:
            details.append(review)
        if any(_exists(bundle_dir / name)
               for name in ("enex", "enex.part")):
            details.append("completion started, not finished")
        candidates.append(Event(
            "p2.open", when_ns, f"P2 opened: {source.name}",
            tuple(details), source.name))
    return _newest(candidates)


def _archived_source(root: Path, bundle_name: str) -> Path | None:
    directory = config.path(root, ["p2", "done", "in"])
    rendered = [bundle_name]
    rendered.extend(f"{bundle_name}{suffix}"
                    for suffix in names.QUEUE_SUFFIXES["p2"])
    matches = []
    for name in dict.fromkeys(rendered):
        path = directory / name
        if path.is_symlink() and not path.exists():
            fs.raise_if_not_file(path)
        if path.is_file():
            matches.append(path)
    if len(matches) != 1:
        return None
    return matches[0]


def _p2_complete(root: Path, recognized) -> Event | None:
    candidates: list[Event] = []
    output = config.path(root, ["p2", "done", "out"])
    for yes in _matching_files(output, "*.p2.yes"):
        bundle_name = yes.name.removesuffix(".p2.yes")
        source = _archived_source(root, bundle_name)
        if source is None:
            continue
        no = output / f"{bundle_name}.p2.no"
        if no.is_symlink() and not no.exists():
            fs.raise_if_not_file(no)
        if no.exists():
            fs.raise_if_not_file(no)
        when_ns = max(yes.stat().st_mtime_ns,
                      no.stat().st_mtime_ns if no.exists() else 0)
        no_count = f"{fs.line_count(no)} pairs" if no.exists() else "not recorded"
        enex = output / "enex" / bundle_name
        _exists(enex)
        if enex.is_dir():
            parts = sum(1 for path in enex.iterdir()
                        if path.is_file() and path.name.endswith(".enex"))
            note_parts = str(parts)
        else:
            note_parts = "not recorded"
        details = [f"source: {source.name} ({fs.line_count(source)} pairs)",
                   f"YES: {fs.line_count(yes)} pairs; NO: {no_count}; "
                   f"note parts: {note_parts}"]
        review = _review_detail(recognized, source)
        if review is not None:
            details.append(review)
        candidates.append(Event(
            "p2.complete", when_ns, f"P2 completed: {bundle_name}",
            tuple(details)))
    return _newest(candidates)


def _classified(root: Path, kind: str) -> Event | None:
    path = fs.optional_file(config.classified(root, kind))
    if path is None or path.stat().st_size == 0:
        return None
    return Event(f"classified.{kind}", path.stat().st_mtime_ns,
                 f"Classified {kind.upper()} changed",
                 (f"current set: {fs.line_count(path)} pairs",))


def _words(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines()
            if line.strip()}


def _dictionary_event(root: Path) -> Event | None:
    tree = dictionary.tree(root)
    reviewed = dictionary.records(tree.reviewed_inputs, dictionary.REVIEWED)
    latest = max(reviewed, key=lambda item: item[0], default=None)
    marker = fs.optional_file(generation.stamp(tree.derived))
    if marker is None:
        marker = fs.optional_file(tree.derived)
    derived = fs.optional_file(tree.derived)
    if latest is None and marker is None:
        return None

    round_ns = latest[1].stat().st_mtime_ns if latest is not None else -1
    marker_ns = marker.stat().st_mtime_ns if marker is not None else -1
    unconfirmed = latest is not None and round_ns > marker_ns
    details: list[str] = []
    if derived is not None:
        details.append(f"current dictionary: {fs.line_count(derived)} words")
    else:
        details.append("current dictionary: not available")

    if latest is not None:
        ordinal, reviewed_path = latest
        enex = tree.enex_archives / reviewed_path.name
        _exists(enex)
        kind = "review" if enex.is_dir() else "direct removal"
        details.append(
            f"latest round: {ordinal} ({kind}), "
            f"{fs.line_count(reviewed_path)} reviewed words")
        suffix = f".{dictionary.REVIEWED}.{ordinal}"
        stem = reviewed_path.name.removesuffix(suffix)
        removal = tree.removals / f"{stem}.{dictionary.REMOVED}.{ordinal}"
        if removal.is_symlink() and not removal.exists():
            fs.raise_if_not_file(removal)
        if removal.exists():
            fs.raise_if_not_file(removal)
            detail = f"removals: {fs.line_count(removal)} words"
            base = fs.optional_file(tree.base)
            if base is not None and derived is not None:
                impact = len(_words(removal) & (_words(base) - _words(derived)))
                detail += f"; current impact: {impact} words"
            details.append(detail)
        else:
            details.append("removal record was deleted")

    reference_ns = marker_ns
    removal_times = [tree.removals.stat().st_mtime_ns]
    for _, path in dictionary.records(tree.removals, dictionary.REMOVED):
        if path.is_symlink() and not path.exists():
            fs.raise_if_not_file(path)
        removal_times.append(path.stat().st_mtime_ns)
    if reference_ns >= 0 and max(removal_times) > reference_ns:
        details.append("removal records changed since the last rebuild")

    if unconfirmed:
        assert latest is not None
        ordinal = latest[0]
        title = f"Dictionary round {ordinal} recorded; publication is not confirmed"
        return Event("dictionary", round_ns, title, tuple(details))
    assert marker is not None
    return Event("dictionary", marker_ns,
                 "Dictionary rebuilt", tuple(details))


def _word_count(path: Path) -> int:
    return len(dictionary.read_words(path))


def _dictionary_queue(root: Path) -> Event | None:
    candidates: list[Event] = []
    queued = config.path(root, ["dict", "queued"])
    for source in _matching_files(queued, "*"):
        try:
            count = _word_count(source)
        except ValueError:
            continue
        candidates.append(Event(
            "dictionary.queue", source.stat().st_mtime_ns,
            f"Dictionary review queued: {source.name}",
            (f"source: {count} words",), source.name))
    evaluating = config.path(root, ["dict", "eval"])
    for bundle_dir in sorted(path for path in evaluating.iterdir()
                             if path.is_dir()):
        ctx = Context(root=root, phase="dict", bundle_name=bundle_dir.name)
        open_source = _source(ctx)
        if open_source is None:
            continue
        try:
            count = _word_count(open_source)
        except ValueError:
            continue
        candidates.append(Event(
            "dictionary.queue", open_source.stat().st_mtime_ns,
            f"Dictionary review queued: {open_source.name}",
            (f"source: {count} words",), open_source.name))
    return _newest(candidates)


def _dictionary_open(root: Path) -> Event | None:
    candidates: list[Event] = []
    evaluating = config.path(root, ["dict", "eval"])
    for bundle_dir in sorted(path for path in evaluating.iterdir()
                             if path.is_dir()):
        ctx = Context(root=root, phase="dict", bundle_name=bundle_dir.name)
        source = _source(ctx)
        if source is None:
            continue
        opened = _opening_time(bundle_dir, source)
        if opened is None:
            continue
        when_ns, evaluated = opened
        try:
            source_count = _word_count(source)
            evaluated_count = _word_count(evaluated)
        except ValueError:
            continue
        details = [f"source: {source_count} words"]
        if evaluated != source:
            details.append(f"filtered: {evaluated_count} words")
        if any(_exists(bundle_dir / name)
               for name in ("enex", "enex.part")):
            details.append("completion started, not finished")
        candidates.append(Event(
            "dictionary.open", when_ns,
            f"Dictionary review opened: {source.name}",
            tuple(details), source.name))
    return _newest(candidates)


def _best_file(target: state.Target, name: str, getter, *,
               skip_broken: bool = True) -> tuple[Path, Path] | None:
    path = target.artifact(name)
    if path.is_symlink() and not path.exists():
        if skip_broken:
            return None
        fs.raise_if_not_file(path)
    found = getter()
    if found is None:
        return None
    referent = found.resolve(strict=True)
    fs.raise_if_not_file(referent)
    return found, referent


def _best_events(targets: list[state.Target]) -> list[Event]:
    events: list[Event] = []
    for target in targets:
        for name, getter in (
                ("best.pairs", lambda t=target: fs.optional_file(
                    t.artifact("best.pairs"))),
                ("no.pairs", lambda t=target: state.target_no_pairs(t))):
            found = _best_file(target, name, getter)
            if found is None:
                continue
            _, referent = found
            events.append(Event(
                f"best.{target.address}.{name}", referent.stat().st_mtime_ns,
                f"BEST {target.address} {name} changed",
                (f"current set: {fs.line_count(referent)} pairs",)))

        seed = _best_file(
            target, "dfs.seed",
            lambda t=target: fs.optional_file(t.artifact("dfs.seed")))
        if seed is not None and seed[0].is_symlink():
            _, results = seed
            fs.raise_if_not_readable(results)
            events.append(Event(
                f"best.{target.address}.dfs.seed",
                results.stat().st_mtime_ns,
                f"BEST {target.address} dfs.seed published",
                (f"results: {fs.line_count(results)} rows",)))

        best = _best_file(
            target, "dfs.best",
            lambda t=target: fs.optional_file(t.artifact("dfs.best")))
        pairs = _best_file(
            target, "dfs.best.pairs",
            lambda t=target: fs.optional_file(t.artifact("dfs.best.pairs")),
            skip_broken=False)
        if (best is not None and best[0].is_symlink()
                and pairs is not None):
            _, results = best
            _, searched = pairs
            fs.raise_if_not_readable(results)
            fs.raise_if_not_readable(searched)
            events.append(Event(
                f"best.{target.address}.dfs.best",
                searched.stat().st_mtime_ns,
                f"BEST {target.address} dfs.best published",
                (f"results: {fs.line_count(results)} rows",
                 f"search pairs: {fs.line_count(searched)} pairs")))

        top = _best_file(
            target, "top.segments",
            lambda t=target: fs.optional_file(t.artifact("top.segments")))
        if top is not None:
            path, content = top
            marker = generation.stamp(path)
            if marker.is_symlink() and not marker.exists():
                fs.raise_if_not_file(marker)
            clock = fs.optional_file(marker) or content
            source = state.top_segments_source(target)
            details = [f"rows: {fs.line_count(content)}",
                       f"source: {source}"]
            content_ns = content.stat().st_mtime_ns
            clock_ns = clock.stat().st_mtime_ns
            if content_ns < clock_ns:
                details.append(f"content last changed: {_time(content_ns)}")
            events.append(Event(
                f"best.{target.address}.top.segments", clock_ns,
                f"BEST {target.address} top.segments published",
                tuple(details)))
    return events


def _merge(queue: Event | None, opened: Event | None, label: str) \
        -> list[Event]:
    if queue is None:
        return [] if opened is None else [opened]
    if opened is None or queue.source_name != opened.source_name:
        return [queue] if opened is None else [queue, opened]
    title = f"{label} queued and opened: {opened.source_name}"
    return [replace(opened, domain=f"{queue.domain}+{opened.domain}",
                    title=title)]


def collect(root: Path) -> list[Event]:
    recognized, targets = _best_reviews(root)
    p2_queue = _p2_queue(root, recognized)
    p2_open = _p2_open(root, recognized)
    dict_queue = _dictionary_queue(root)
    dict_open = _dictionary_open(root)
    events = [
        _p2_complete(root, recognized),
        _classified(root, "yes"),
        _classified(root, "no"),
        _dictionary_event(root),
        *_merge(p2_queue, p2_open, "P2"),
        *_merge(dict_queue, dict_open, "Dictionary review"),
        *_best_events(targets),
    ]
    return sorted((event for event in events if event is not None),
                  key=lambda event: (-event.when_ns, event.domain,
                                     event.title))


def render(events: list[Event], count: int) -> None:
    selected = events[:count]
    if not selected:
        print("No workflow history.")
        return
    for event in selected:
        print(f"{_time(event.when_ns)}  {event.title}")
        for detail in event.details:
            print(f"  {detail}")


class History(command.Action):
    def __init__(self):
        super().__init__(
            summary="history  — show recent workflow milestones (default: 5)",
            positional="[COUNT]")

    def run(self, command_text, opts, argv) -> int:
        if len(argv) > 1:
            return usage.invalid_argument(argv[1],
                                          self.format_help(command_text))
        try:
            count = DEFAULT_COUNT if not argv else int(argv[0])
        except ValueError:
            return usage.invalid_argument(argv[0],
                                          self.format_help(command_text))
        if count <= 0:
            return usage.invalid_argument(argv[0],
                                          self.format_help(command_text))
        render(collect(opts.dir), count)
        return 0


COMMAND = History()
