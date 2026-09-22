"""Shared cluedata index format, builder, and whole-word lookup."""

import json
import mmap
import os
import re
import shutil
import struct
import sys
import tempfile
import uuid
from array import array
from itertools import islice, repeat
from pathlib import Path

import numpy as np


DEFAULT_DATA = Path(__file__).parent / "data" / "cluedata"
DEFAULT_INDEX = Path(__file__).parent / "data" / "index"
WORD = re.compile(rb"[a-z0-9]+")
ADJACENT_QUERY = re.compile(rb"[A-Za-z0-9]+ [A-Za-z0-9]+")
ADJACENT_FILE_QUERY = re.compile(rb"[A-Za-z0-9]+,[A-Za-z0-9]+")
EXACT_QUERY = re.compile(rb"[A-Za-z0-9]+(?: [A-Za-z0-9]+)?")
EXACT_FILE_QUERY = re.compile(rb"[A-Za-z0-9]+(?:,[A-Za-z0-9]+)?")
FORMAT_VERSION = 2
U32_MAX = np.iinfo(np.uint32).max
ADJACENT_CHUNK = 65536


def clean_words(text: bytes) -> list[bytes]:
    """Apply make-index's ASCII word boundaries and apostrophe deletion."""
    return WORD.findall(text.replace(b"'", b"").lower())


def adjacent_words(text: bytes) -> list[tuple[bytes, bytes]]:
    """Return clean_words pairs separated by exactly one space."""
    text = text.replace(b"'", b"").lower()
    pairs = []
    previous = None
    for match in WORD.finditer(text):
        if (previous is not None and match.start() == previous.end() + 1
                and text[previous.end()] == 0x20):
            pairs.append((previous.group(), match.group()))
        previous = match
    return pairs


def bigram_key(first: int, second: int) -> int:
    return first << 32 | second


def read_u32(data: mmap.mmap, offset: int) -> int:
    if offset + 4 > len(data):
        raise ValueError(f"truncated cluedata at byte {offset}")
    return struct.unpack_from("<I", data, offset)[0]


def scan(data: mmap.mmap):
    """Yield (clue number, record offset, clue bytes, refs offset, ref count)."""
    offset = 4
    answer_count = read_u32(data, 0)
    for _ in range(answer_count):
        if offset >= len(data):
            raise ValueError(f"truncated answer at byte {offset}")
        offset += 1 + data[offset]
        if offset > len(data):
            raise ValueError("truncated answer section")
    clue_count = read_u32(data, offset)
    offset += 4
    for clue_id in range(clue_count):
        record_offset = offset
        if offset >= len(data):
            raise ValueError(f"truncated clue at byte {offset}")
        length = data[offset]
        clue = data[offset + 1 : offset + 1 + length]
        offset += 1 + length
        ref_count = read_u32(data, offset)
        offset += 4
        if offset + 4 * ref_count > len(data):
            raise ValueError(f"truncated references at byte {offset}")
        yield clue_id, record_offset, clue, offset, ref_count
        offset += 4 * ref_count


def answers_from_data(data: mmap.mmap) -> list[bytes]:
    offset = 4
    answers = []
    for _ in range(read_u32(data, 0)):
        length = data[offset]
        answers.append(data[offset + 1 : offset + 1 + length])
        offset += 1 + length
    return answers


def build(data_path: Path, index_path: Path, replace: bool = False) -> None:
    if index_path.exists():
        if not replace:
            raise ValueError(f"index already exists: {index_path}; use --replace")
        if (not index_path.is_dir() or index_path.is_symlink()
                or not (index_path / "metadata.json").is_file()):
            raise ValueError(f"refusing to replace a non-index directory: {index_path}")
    source_stat = data_path.stat()
    with data_path.open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        counts: dict[bytes, int] = {}
        clue_count = ref_count = 0
        for clue_id, _, clue, _, nrefs in scan(data):
            clue_count = clue_id + 1
            ref_count += nrefs
            for word in set(clean_words(clue)):
                counts[word] = counts.get(word, 0) + 1

        posting_count = sum(counts.values())
        if max(source_stat.st_size, clue_count, ref_count, posting_count) > U32_MAX:
            raise ValueError("cluedata is too large for uint32 index arrays")

        tokens = sorted(counts)
        word_ids = {word: i for i, word in enumerate(tokens)}
        starts = np.empty(len(tokens) + 1, dtype=np.uint32)
        starts[0] = 0
        np.cumsum(np.fromiter((counts[t] for t in tokens), dtype=np.uint64),
                  out=starts[1:])
        cursors = starts[:-1].copy()

        index_path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{index_path.name}.",
                                        dir=index_path.parent))
        try:
            clue_off = np.lib.format.open_memmap(
                staging / "clue_off.npy", mode="w+", dtype="<u4", shape=(clue_count,)
            )
            ref_start = np.lib.format.open_memmap(
                staging / "ref_start.npy", mode="w+", dtype="<u4",
                shape=(clue_count + 1,)
            )
            refs = np.lib.format.open_memmap(
                staging / "refs.npy", mode="w+", dtype="<u4", shape=(ref_count,)
            )
            postings = np.lib.format.open_memmap(
                staging / "postings.npy", mode="w+", dtype="<u4",
                shape=(posting_count,)
            )
            np.save(staging / "postings_start.npy", starts)
            (staging / "tokens.txt").write_bytes(b"\n".join(tokens) + b"\n")
            (staging / "answers.txt").write_bytes(
                b"\n".join(answers_from_data(data)) + b"\n"
            )

            next_ref = 0
            bigrams = array("Q")
            for clue_id, offset, clue, refs_offset, nrefs in scan(data):
                clue_off[clue_id] = offset
                ref_start[clue_id] = next_ref
                if nrefs:
                    refs[next_ref : next_ref + nrefs] = np.frombuffer(
                        data, dtype="<u4", count=nrefs, offset=refs_offset
                    )
                next_ref += nrefs
                for word in set(clean_words(clue)):
                    word_id = word_ids[word]
                    pos = cursors[word_id]
                    postings[pos] = clue_id
                    cursors[word_id] = pos + 1
                for first, second in adjacent_words(clue):
                    bigrams.append(bigram_key(word_ids[first], word_ids[second]))
            bigrams = np.unique(np.frombuffer(bigrams, dtype=np.uint64))
            np.save(staging / "bigrams.npy", bigrams)
            ref_start[clue_count] = next_ref
            if not np.array_equal(cursors, starts[1:]):
                raise ValueError("postings count changed during build")
            for mapped in (clue_off, ref_start, refs, postings):
                mapped.flush()
            del clue_off, ref_start, refs, postings

            final_stat = data_path.stat()
            if (final_stat.st_size, final_stat.st_mtime_ns) != (
                source_stat.st_size, source_stat.st_mtime_ns
            ):
                raise ValueError("cluedata changed during build")
            metadata = {
                "version": FORMAT_VERSION,
                "source_size": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "clues": clue_count,
                "references": ref_count,
                "postings": posting_count,
                "words": len(tokens),
                "bigrams": len(bigrams),
            }
            (staging / "metadata.json").write_text(json.dumps(metadata) + "\n")
            if index_path.exists():
                backup = index_path.with_name(
                    f".{index_path.name}.old-{uuid.uuid4().hex}"
                )
                os.rename(index_path, backup)
                try:
                    os.rename(staging, index_path)
                except OSError:
                    os.rename(backup, index_path)
                    raise
                shutil.rmtree(backup)
            else:
                os.rename(staging, index_path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def intersect(small: np.ndarray, big: np.ndarray) -> np.ndarray:
    if not len(small) or not len(big):
        return small[:0]
    pos = np.searchsorted(big, small)
    valid = pos < len(big)
    return small[valid & (big[np.minimum(pos, len(big) - 1)] == small)]


def in_sorted(keys: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Return a mask of which keys occur in the sorted array table."""
    # Looking up sorted keys walks table in order, which is much faster
    # than random lookups into a large table.
    order = np.argsort(keys)
    sorted_keys = keys[order]
    found = np.zeros(len(keys), dtype=bool)
    if len(table):
        pos = np.minimum(np.searchsorted(table, sorted_keys), len(table) - 1)
        found[order] = table[pos] == sorted_keys
    return found


def adjacent_ids(queries: list[bytes], word_ids: dict[bytes, int]) -> np.ndarray:
    """Return an (n, 2) array of word ids for "first,second" query lines.

    An id is -1 for an unknown word, and both are -1 unless the line has
    exactly one comma.
    """
    text = b"\n".join(queries).lower()
    separators = np.frombuffer(text, dtype=np.uint8)
    separators = separators[(separators == 0x2C) | (separators == 0x0A)]
    if (len(separators) == 2 * len(queries) - 1
            and (separators[0::2] == 0x2C).all()):
        # Every line has exactly one comma, so one flat split lines up.
        words = text.replace(b"\n", b",").split(b",")
        return np.fromiter(map(word_ids.get, words, repeat(-1)),
                           dtype=np.int64, count=len(words)).reshape(-1, 2)
    ids = []
    for query_line in queries:
        parts = query_line.lower().split(b",")
        ids.append((word_ids.get(parts[0], -1), word_ids.get(parts[1], -1))
                   if len(parts) == 2 else (-1, -1))
    return np.array(ids, dtype=np.int64).reshape(-1, 2)


def print_adjacent_queries(source, word_ids: dict[bytes, int],
                           bigrams: np.ndarray, forward: bool) -> None:
    """Print -f -j query lines whose two words are adjacent in some clue.

    Lines are checked a chunk at a time against the sorted bigram keys.
    """
    while lines := list(islice(source, ADJACENT_CHUNK)):
        queries = [line.rstrip(b"\r\n") for line in lines]
        ids = adjacent_ids(queries, word_ids)
        known = (ids >= 0).all(axis=1)
        # Known words are all [a-z0-9]+, so the regex check is only needed
        # for lines with an unknown part.
        end = len(queries)
        for i in np.flatnonzero(~known):
            if ADJACENT_FILE_QUERY.fullmatch(queries[i]) is None:
                end = i
                break
        first = ids[:, 0].astype(np.uint64)
        second = ids[:, 1].astype(np.uint64)
        keys = first << np.uint64(32) | second
        if not forward:
            keys = np.concatenate((keys, second << np.uint64(32) | first))
        found = known & in_sorted(keys, bigrams).reshape(-1, len(queries)).any(axis=0)
        matches = [queries[i] for i in np.flatnonzero(found[:end])]
        if matches:
            print(b"\n".join(matches).decode("utf-8", errors="replace"))
        if end < len(queries):
            raise ValueError(
                "two words required for --adjacent: "
                f"{queries[end].decode('utf-8', errors='replace')!r}"
            )


def clue_word_index(postings: np.ndarray, starts: list[int],
                    clue_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Invert postings into (clue_start, words): each clue's word ids."""
    word_ids = np.repeat(np.arange(len(starts) - 1, dtype=np.uint32),
                         np.diff(starts))
    words = word_ids[np.argsort(postings, kind="stable")]
    clue_start = np.zeros(clue_count + 1, dtype=np.int64)
    np.cumsum(np.bincount(postings, minlength=clue_count), out=clue_start[1:])
    return clue_start, words


def neighbors(word_id: int, starts: list[int], postings: np.ndarray,
              clue_start: np.ndarray, words: np.ndarray) -> bytes:
    """Return a per-word-id mask of words sharing at least one clue with word_id."""
    clues = postings[starts[word_id] : starts[word_id + 1]]
    begin = clue_start[clues]
    lengths = clue_start[clues + 1] - begin
    positions = (np.arange(lengths.sum())
                 + np.repeat(begin - (np.cumsum(lengths) - lengths), lengths))
    mask = np.zeros(len(starts) - 1, dtype=np.uint8)
    mask[words[positions]] = 1
    return mask.tobytes()


def validate_index(data_path: Path, index_path: Path) -> None:
    metadata = json.loads((index_path / "metadata.json").read_text())
    source_stat = data_path.stat()
    if metadata["version"] != FORMAT_VERSION:
        raise ValueError("unsupported index version; rebuild the index")
    if (metadata["source_size"], metadata["source_mtime_ns"]) != (
        source_stat.st_size, source_stat.st_mtime_ns
    ):
        raise ValueError("index does not match cluedata; rebuild the index")


def query(data_path: Path, index_path: Path, input_path: str | None,
          query_text: str | None,
          json_output: bool = False, show_results: bool = False,
          adjacent: bool = False, exact: bool = False,
          forward: bool = False, sorted_input: bool = False) -> None:
    if (adjacent and input_path is None
            and ADJACENT_QUERY.fullmatch(query_text.encode("utf-8")) is None):
        raise ValueError(
            f"two words required for --adjacent: {query_text!r}"
        )
    if (exact and input_path is None
            and EXACT_QUERY.fullmatch(query_text.encode("utf-8")) is None):
        raise ValueError("one or two words required for --exact")
    validate_index(data_path, index_path)
    tokens = (index_path / "tokens.txt").read_bytes().splitlines()
    word_ids = {word: i for i, word in enumerate(tokens)}
    answers = (index_path / "answers.txt").read_bytes().splitlines()
    # Python ints for scalar reads; plain ndarray views skip np.memmap's
    # per-index overhead while still reading the mapped pages.
    starts = np.load(index_path / "postings_start.npy").tolist()
    postings = np.asarray(np.load(index_path / "postings.npy", mmap_mode="r"))
    clue_off = np.asarray(np.load(index_path / "clue_off.npy", mmap_mode="r"))
    ref_start = np.asarray(np.load(index_path / "ref_start.npy", mmap_mode="r"))
    refs = np.asarray(np.load(index_path / "refs.npy", mmap_mode="r"))
    chunked = adjacent and input_path is not None and not show_results
    bigrams = (set(np.load(index_path / "bigrams.npy").tolist())
               if adjacent and not chunked else set())
    query_only = (input_path is not None and not show_results
                  and not (adjacent or exact or forward))
    clue_index = None
    lead_id = None
    lead_neighbors = b""

    if input_path is None:
        source = (query_text.encode("utf-8"),)
    else:
        source = sys.stdin.buffer if input_path == "-" else open(input_path, "rb")
    try:
        if chunked:
            print_adjacent_queries(source, word_ids,
                                   np.load(index_path / "bigrams.npy"), forward)
            return
        with data_path.open("rb") as stream, mmap.mmap(
            stream.fileno(), 0, access=mmap.ACCESS_READ
        ) as data:
            for line in source:
                raw_query = line.rstrip(b"\r\n")
                if adjacent or exact:
                    separator = b" " if input_path is None else b","
                    parts = raw_query.lower().split(separator)
                    # Known words are all [a-z0-9]+, so the regex check is
                    # only needed for lines with an unknown part.
                    if not ((len(parts) == 2 or exact and len(parts) == 1)
                            and all(part in word_ids for part in parts)):
                        if exact:
                            pattern = (EXACT_QUERY if input_path is None
                                       else EXACT_FILE_QUERY)
                            error = "one or two words required for --exact"
                        else:
                            pattern = (ADJACENT_QUERY if input_path is None
                                       else ADJACENT_FILE_QUERY)
                            error = (
                                "two words required for --adjacent: "
                                f"{raw_query.decode('utf-8', errors='replace')!r}"
                            )
                        if pattern.fullmatch(raw_query) is None:
                            raise ValueError(error)
                        continue
                    words = set(parts)
                    lead_word = parts[0]
                    if adjacent:
                        phrases = {(parts[0], parts[1])}
                        if not forward:
                            phrases.add((parts[1], parts[0]))
                    elif len(parts) == 1:
                        phrases = (parts[0],)
                    else:
                        phrases = (parts[0] + b" " + parts[1],)
                        if not forward:
                            phrases += (parts[1] + b" " + parts[0],)
                elif input_path is None:
                    ordered_words = [word for word in
                                     raw_query.lower().replace(b"'", b"").split(b" ")
                                     if word]
                    words = set(ordered_words)
                else:
                    # A plain split gives the same words as clean_words when
                    # every part is a known word; otherwise use the regex.
                    ordered_words = raw_query.lower().split(b",")
                    if not all(word in word_ids for word in ordered_words):
                        ordered_words = clean_words(raw_query)
                    words = set(ordered_words)
                    lead_word = ordered_words[0] if ordered_words else None
                if not words or any(word not in word_ids for word in words):
                    continue
                if adjacent:
                    if not any(bigram_key(word_ids[first], word_ids[second])
                               in bigrams for first, second in phrases):
                        continue
                elif sorted_input and len(words) == 2:
                    # Consecutive lines usually share a lead word, so its
                    # co-occurring words are computed once and reused.
                    if clue_index is None:
                        clue_index = clue_word_index(postings, starts,
                                                     len(clue_off))
                    if word_ids[lead_word] != lead_id:
                        lead_id = word_ids[lead_word]
                        lead_neighbors = neighbors(lead_id, starts, postings,
                                                   *clue_index)
                    other = next(word for word in words if word != lead_word)
                    if not lead_neighbors[word_ids[other]]:
                        continue
                    if query_only:
                        print(raw_query.decode("utf-8", errors="replace"))
                        continue
                ids = sorted((word_ids[word] for word in words),
                             key=lambda i: starts[i + 1] - starts[i])
                first = ids[0]
                matches = postings[starts[first] : starts[first + 1]]
                for word_id in ids[1:]:
                    matches = intersect(
                        matches, postings[starts[word_id] : starts[word_id + 1]]
                    )
                    if not len(matches):
                        break
                if not len(matches):
                    continue
                if query_only:
                    print(raw_query.decode("utf-8", errors="replace"))
                    continue
                printed_query = False
                for clue_id in matches:
                    clue_id = int(clue_id)
                    offset = int(clue_off[clue_id])
                    clue = data[offset + 1 : offset + 1 + data[offset]]
                    if exact:
                        if clue.lower() not in phrases:
                            continue
                    elif adjacent:
                        if phrases.isdisjoint(adjacent_words(clue)):
                            continue
                    elif forward:
                        clue_words = iter(clean_words(clue))
                        if not all(word in clue_words for word in ordered_words):
                            continue
                    if input_path is not None and not printed_query:
                        print(raw_query.decode("utf-8", errors="replace"))
                        printed_query = True
                    if input_path is not None and not show_results:
                        break
                    decoded_refs = [
                        {"answer": answers[int(ref) >> 1].decode("latin-1"),
                         "bit": int(ref) & 1}
                        for ref in refs[ref_start[clue_id] : ref_start[clue_id + 1]]
                    ]
                    clue_text = clue.decode("latin-1")
                    if json_output:
                        print(json.dumps({
                            "query": raw_query.decode("utf-8", errors="replace"),
                            "offset": offset,
                            "clue": clue_text,
                            "answers": decoded_refs,
                        }, ensure_ascii=False))
                    else:
                        answer_text = ", ".join(ref["answer"] for ref in decoded_refs)
                        print(f"0x{offset:x} {clue_text!r} -> {answer_text}")
    finally:
        if input_path not in (None, "-"):
            source.close()


def query_answer(data_path: Path, index_path: Path, answer_text: str,
                 json_output: bool = False) -> None:
    validate_index(data_path, index_path)
    answers = (index_path / "answers.txt").read_bytes().splitlines()
    needle = answer_text.encode("utf-8").upper()
    answer_ids = [i for i, answer in enumerate(answers) if answer.upper() == needle]
    if not answer_ids:
        return

    clue_off = np.load(index_path / "clue_off.npy", mmap_mode="r")
    ref_start = np.load(index_path / "ref_start.npy", mmap_mode="r")
    refs = np.load(index_path / "refs.npy", mmap_mode="r")
    positions = np.flatnonzero(np.isin(refs >> 1, answer_ids))
    clue_ids = np.unique(np.searchsorted(ref_start, positions, side="right") - 1)
    answer_id_set = set(answer_ids)

    with data_path.open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        for clue_id in clue_ids:
            clue_id = int(clue_id)
            offset = int(clue_off[clue_id])
            clue = data[offset + 1 : offset + 1 + data[offset]].decode("latin-1")
            decoded_refs = [
                {"answer": answers[int(ref) >> 1].decode("latin-1"),
                 "bit": int(ref) & 1}
                for ref in refs[ref_start[clue_id] : ref_start[clue_id + 1]]
                if int(ref) >> 1 in answer_id_set
            ]
            if json_output:
                print(json.dumps({
                    "query": answer_text,
                    "offset": offset,
                    "clue": clue,
                    "answers": decoded_refs,
                }, ensure_ascii=False))
            else:
                answer_names = ", ".join(ref["answer"] for ref in decoded_refs)
                print(f"0x{offset:x} {clue!r} -> {answer_names}")
