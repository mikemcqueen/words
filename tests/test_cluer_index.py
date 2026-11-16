"""Small binary fixture for the cluedata whole-word index."""

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "cluer"


def cluedata(answers, clues):
    data = bytearray(struct.pack("<I", len(answers)))
    for answer in answers:
        value = answer.encode("ascii")
        data.extend(bytes([len(value)]) + value)
    data.extend(struct.pack("<I", len(clues)))
    for clue, references in clues:
        value = clue.encode("latin-1")
        data.extend(bytes([len(value)]) + value)
        data.extend(struct.pack("<I", len(references)))
        for reference in references:
            data.extend(struct.pack("<I", reference))
    return data


class CluerIndexTests(unittest.TestCase):
    def run_tool(self, tool, *args):
        return subprocess.run([sys.executable, str(TOOLS / tool), *map(str, args)],
                              text=True, capture_output=True)

    def test_exact_clue_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "cluedata"
            index = root / "index"
            terms = root / "terms.txt"
            data.write_bytes(cluedata(
                ["APPLE"],
                [(clue, [0]) for clue in (
                    "New", "NEW", "New York", "York New", "New York!",
                    "A New York", "New-York",
                )],
            ))
            result = self.run_tool("build_index.py", "--data", data,
                                   "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)

            def exact(*args):
                return self.run_tool("query_index.py", "-e", *args,
                                     "--data", data, "--index", index,
                                     "--json")

            result = exact("new")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([json.loads(line)["clue"] for line in
                              result.stdout.splitlines()], ["New", "NEW"])

            result = exact("new york")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([json.loads(line)["clue"] for line in
                              result.stdout.splitlines()],
                             ["New York", "York New", "New-York"])

            result = self.run_tool("query_index.py", "-j", "new york",
                                   "--data", data, "--index", index,
                                   "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([json.loads(line)["clue"] for line in
                              result.stdout.splitlines()],
                             ["New York", "York New", "New York!", "A New York",
                              "New-York"])

            terms.write_text("new\nnew,york\nyork,new\nmissing\n")
            result = exact("-f", terms, "-r")
            self.assertEqual(result.returncode, 0, result.stderr)
            output = result.stdout.splitlines()
            self.assertEqual([line for line in output if not line.startswith("{")],
                             ["new", "new,york", "york,new"])
            self.assertEqual([(hit["query"], hit["clue"]) for hit in
                              (json.loads(line) for line in output
                               if line.startswith("{"))], [
                ("new", "New"), ("new", "NEW"),
                ("new,york", "New York"), ("new,york", "York New"),
                ("new,york", "New-York"),
                ("york,new", "New York"), ("york,new", "York New"),
                ("york,new", "New-York"),
            ])

            result = exact("-f", terms, "-v")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["missing"])

            for query in ("", "new york city", "new,york", "new-york"):
                result = exact(query)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("one or two words required for --exact",
                              result.stderr)

            terms.write_text("new,york,city\n")
            result = exact("-f", terms)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("one or two words required for --exact",
                          result.stderr)

            result = exact("-a", "APPLE")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--exact cannot be used with -a/--answer",
                          result.stderr)

            result = exact("-j", "new york")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--exact cannot be used with -j/--adjacent",
                          result.stderr)

    def test_adjacent_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "cluedata"
            index = root / "index"
            terms = root / "terms.txt"
            data.write_bytes(cluedata(
                ["APPLE"],
                [(clue, [0]) for clue in (
                    "Firewood wood", "Don't stop", "Rock 'n' roll",
                    "New  York", "Stop now", "wood",
                    '"Blue Teal" lamp',
                )],
            ))
            result = self.run_tool("build_index.py", "--data", data,
                                   "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)

            def adjacent(*args):
                return self.run_tool("query_index.py", "-j", *args,
                                     "--data", data, "--index", index)

            result = adjacent("dont stop")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([line.split(" ", 1)[1] for line in
                              result.stdout.splitlines()],
                             ["\"Don't stop\" -> APPLE"])

            terms.write_text(
                "wood,wood\nfirewood,wood\nstop,dont\nn,roll\nnew,york\n"
                "now,stop\nteal,lamp\n"
            )
            result = adjacent("-f", terms)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             ["firewood,wood", "stop,dont", "n,roll", "new,york",
                              "now,stop", "teal,lamp"])

            for threads in ("2", "0"):
                result = adjacent("-f", terms, "--threads", threads)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(),
                                 ["firewood,wood", "stop,dont", "n,roll",
                                  "new,york", "now,stop", "teal,lamp"])

            result = adjacent("-f", terms, "--forward")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             ["firewood,wood", "n,roll", "new,york",
                              "teal,lamp"])

            for threads in ("1", "2"):
                result = adjacent("-f", terms, "-v", "--threads", threads)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["wood,wood"])

            result = adjacent("-f", terms, "-v", "--forward")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             ["wood,wood", "stop,dont", "now,stop"])

            bad_terms = root / "bad_terms.txt"
            bad_terms.write_text("firewood,wood\nnew york\nn,roll\n")
            result = adjacent("-f", bad_terms, "--threads", "2")
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout.splitlines(), ["firewood,wood"])
            self.assertIn("two words required for --adjacent: 'new york'",
                          result.stderr)

            # Every line has one comma, but one line is still malformed.
            for bad, shown in ((b"n,", "n,"), (b",roll", ",roll"),
                               (b"n-x,roll", "n-x,roll"),
                               (b"caf\xc3\xa9,roll", "café,roll")):
                bad_terms.write_bytes(b"firewood,wood\n" + bad + b"\nn,roll\n")
                result = adjacent("-f", bad_terms)
                self.assertEqual(result.returncode, 1, bad)
                self.assertEqual(result.stdout.splitlines(), ["firewood,wood"])
                self.assertIn(f"two words required for --adjacent: {shown!r}",
                              result.stderr)

            result = adjacent("-f", terms, "-r", "--forward")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([line.split(" ", 1)[-1] for line in
                              result.stdout.splitlines()], [
                "firewood,wood",
                "'Firewood wood' -> APPLE",
                "n,roll",
                "\"Rock 'n' roll\" -> APPLE",
                "new,york",
                "'New  York' -> APPLE",
                "teal,lamp",
                "'\"Blue Teal\" lamp' -> APPLE",
            ])

    def test_adjacent_phrase_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "cluedata"
            index = root / "index"
            terms = root / "terms.txt"
            data.write_bytes(cluedata(
                ["APPLE"],
                [(clue, [0]) for clue in (
                    "New York City", "City of New York", "New city York",
                    "The New York Times",
                )],
            ))
            result = self.run_tool("build_index.py", "--data", data,
                                   "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)

            def adjacent(*args):
                return self.run_tool("query_index.py", "-j", *args,
                                     "--data", data, "--index", index)

            def clues(result):
                self.assertEqual(result.returncode, 0, result.stderr)
                return [line.split(" ", 1)[1] for line in
                        result.stdout.splitlines()]

            self.assertEqual(clues(adjacent("new york,city")),
                             ["'New York City' -> APPLE"])
            self.assertEqual(clues(adjacent("City,New York")),
                             ["'New York City' -> APPLE"])
            self.assertEqual(clues(adjacent("city,new york", "--forward")), [])
            self.assertEqual(clues(adjacent("york,new")),
                             ["'New York City' -> APPLE",
                              "'City of New York' -> APPLE",
                              "'The New York Times' -> APPLE"])
            for query in ("new york city", "new york,", "new  york,city",
                          "new york,city,times", "new-york,city"):
                result = adjacent(query)
                self.assertEqual(result.returncode, 1, query)
                self.assertIn(f"two words required for --adjacent: {query!r}",
                              result.stderr)

            # "the new city" has the bigrams "the new" and "new city", but
            # in different clues.
            terms.write_text(
                "new york,city\ncity,new york\nyork,new\nnew york,times\n"
                "new,york times\nyork city,new\nthe new,city\n"
                "new york,zzz\nof,new york\n"
            )
            matches = ["new york,city", "city,new york", "york,new",
                       "new york,times", "new,york times", "york city,new",
                       "of,new york"]
            for threads in ("1", "2"):
                result = adjacent("-f", terms, "--threads", threads)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), matches)

                result = adjacent("-f", terms, "-v", "--threads", threads)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(),
                                 ["the new,city", "new york,zzz"])

            result = adjacent("-f", terms, "--forward")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             ["new york,city", "new york,times",
                              "new,york times", "of,new york"])

            result = adjacent("-f", terms, "-r", "--forward")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([line.split(" ", 1)[1] if line.startswith("0x")
                              else line for line in
                              result.stdout.splitlines()], [
                "new york,city", "'New York City' -> APPLE",
                "new york,times", "'The New York Times' -> APPLE",
                "new,york times", "'The New York Times' -> APPLE",
                "of,new york", "'City of New York' -> APPLE",
            ])

            bad_terms = root / "bad_terms.txt"
            for bad in ("new  york,city", "new york ,city", " new,city",
                        "new york,city,times", "new york city"):
                bad_terms.write_text(f"of,new york\n{bad}\nnew york,city\n")
                for args in ((), ("-r",)):
                    result = adjacent("-f", bad_terms, *args)
                    self.assertEqual(result.returncode, 1, bad)
                    self.assertEqual(result.stdout.splitlines()[:1],
                                     ["of,new york"])
                    self.assertNotIn("new york,city", result.stdout)
                    self.assertIn(
                        f"two words required for --adjacent: {bad!r}",
                        result.stderr)

    def test_build_and_stream_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "cluedata"
            index = root / "index"
            terms = root / "terms.txt"
            data.write_bytes(cluedata(
                ["APPLE", "PEAR", "PLUM"],
                [("Don't stop, New-York!", [1, 2]),
                 ("York's new idea", [4]),
                 ("A Château clue", [0]),
                 ("New York", [2])],
            ))
            terms.write_bytes(
                b"york,new\nyorks,new\nDon't,stop\nnew,new\nart,missing\nch,teau\n___,???\n"
            )
            result = self.run_tool("build_index.py", "--data", data,
                                   "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)

            result = self.run_tool("query_index.py", "-f", terms, "--data", data,
                                   "--index", index, "--json", "-r")
            self.assertEqual(result.returncode, 0, result.stderr)
            output = result.stdout.splitlines()
            self.assertEqual([line for line in output if not line.startswith("{")],
                             ["york,new", "yorks,new", "Don't,stop",
                              "new,new", "ch,teau"])
            hits = [json.loads(line) for line in output if line.startswith("{")]
            self.assertEqual([(hit["query"], hit["clue"]) for hit in hits], [
                ("york,new", "Don't stop, New-York!"),
                ("york,new", "New York"),
                ("yorks,new", "York's new idea"),
                ("Don't,stop", "Don't stop, New-York!"),
                ("new,new", "Don't stop, New-York!"),
                ("new,new", "York's new idea"),
                ("new,new", "New York"),
                ("ch,teau", "A Château clue"),
            ])
            self.assertEqual(hits[0]["answers"], [
                {"answer": "APPLE", "bit": 1},
                {"answer": "PEAR", "bit": 0},
            ])
            self.assertIsInstance(hits[0]["offset"], int)

            pairs = root / "pairs.txt"
            pairs.write_text("new,york\nmissing,word\nch,teau\n")
            result = self.run_tool("query_index.py", "-f", pairs, "--data", data,
                                   "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["new,york", "ch,teau"])

            for extra in ((), ("--forward",), ("--sorted-input",)):
                result = self.run_tool("query_index.py", "-f", pairs, "-v",
                                       *extra, "--data", data, "--index", index)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["missing,word"])

            result = self.run_tool("query_index.py", "-f", terms, "-v",
                                   "--data", data, "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             ["art,missing", "___,???"])

            result = self.run_tool("query_index.py", "-f", pairs, "-v", "-r",
                                   "--data", data, "--index", index)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")

            result = self.run_tool("query_index.py", "new york", "-v",
                                   "--data", data, "--index", index)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")

            result = self.run_tool("query_index.py", "-f", pairs, "-r",
                                   "--data", data, "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [
                "new,york",
                f"0x{hits[0]['offset']:x} \"Don't stop, New-York!\" -> APPLE, PEAR",
                f"0x{hits[1]['offset']:x} 'New York' -> PEAR",
                "ch,teau",
                f"0x{hits[-1]['offset']:x} 'A Château clue' -> APPLE",
            ])

            result = self.run_tool("query_index.py", "--file", pairs,
                                   "--data", data,
                                   "--index", index, "--json", "--results")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines()[0], "new,york")
            self.assertEqual([hit["clue"] for hit in map(json.loads,
                              [line for line in result.stdout.splitlines()
                               if line.startswith("{")])],
                             ["Don't stop, New-York!", "New York",
                              "A Château clue"])

            result = self.run_tool("query_index.py", "new york", "--data", data,
                                   "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [
                f"0x{hits[0]['offset']:x} \"Don't stop, New-York!\" -> APPLE, PEAR",
                f"0x{hits[1]['offset']:x} 'New York' -> PEAR",
            ])

            result = self.run_tool("query_index.py", "new york", "--data", data,
                                   "--index", index, "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([hit["query"] for hit in map(json.loads,
                              result.stdout.splitlines())],
                             ["new york", "new york"])

            result = self.run_tool("query_index.py", "new,york", "--data", data,
                                   "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")

            result = self.run_tool("query_index.py", "-a", "pear", "--data",
                                   data, "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [
                f"0x{hits[0]['offset']:x} \"Don't stop, New-York!\" -> PEAR",
                f"0x{hits[1]['offset']:x} 'New York' -> PEAR",
            ])

            result = self.run_tool("query_index.py", "--answer", "APPLE",
                                   "--data", data, "--index", index, "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([json.loads(line)["answers"] for line in
                              result.stdout.splitlines()],
                             [[{"answer": "APPLE", "bit": 1}],
                              [{"answer": "APPLE", "bit": 0}]])

            result = self.run_tool("query_index.py", "-a", "MISSING", "--data",
                                   data, "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")

            result = self.run_tool("query_index.py", "new york", "-a", "PEAR",
                                   "--data", data, "--index", index)
            self.assertEqual(result.returncode, 2)
            self.assertIn("provide exactly one", result.stderr)

            result = self.run_tool("query_index.py", "new york", "-r",
                                   "--data", data, "--index", index)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--results requires -f/--file", result.stderr)

            result = self.run_tool("build_index.py", "--data", data,
                                   "--index", index)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)

            result = self.run_tool("build_index.py", "--data", data,
                                   "--index", index, "--replace")
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.run_tool("query_index.py", "-f", terms, "--data", data,
                                   "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)

            data.write_bytes(data.read_bytes() + b"metadata")
            result = self.run_tool("query_index.py", "-f", terms, "--data", data,
                                   "--index", index)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match cluedata", result.stderr)

    def test_letter_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "cluedata"
            index = root / "index"
            words = root / "words.txt"
            data.write_bytes(cluedata(
                ["APPLE"],
                [(clue, [0]) for clue in (
                    "New York", "York minster", "Big apple", "Big big",
                )],
            ))
            words.write_text("new\nyork\nbig\napple\nminster\n")
            result = self.run_tool("build_index.py", "--data", data,
                                   "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.run_tool("query_index.py", "--letters",
                                   "newyorkbigapple", "--dict", words,
                                   "--data", data, "--index", index)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "apple,big\nnew,york\n")


if __name__ == "__main__":
    unittest.main()
