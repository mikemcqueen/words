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
                              result.stdout.splitlines()], ["New York", "York New"])

            result = self.run_tool("query_index.py", "-j", "new york",
                                   "--data", data, "--index", index,
                                   "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([json.loads(line)["clue"] for line in
                              result.stdout.splitlines()],
                             ["New York", "York New", "New York!", "A New York"])

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
                ("york,new", "New York"), ("york,new", "York New"),
            ])

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
                "now,stop\n"
            )
            result = adjacent("-f", terms)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             ["firewood,wood", "stop,dont", "n,roll", "now,stop"])

            result = adjacent("-f", terms, "--forward")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             ["firewood,wood", "n,roll"])

            result = adjacent("-f", terms, "-r", "--forward")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([line.split(" ", 1)[-1] for line in
                              result.stdout.splitlines()], [
                "firewood,wood",
                "'Firewood wood' -> APPLE",
                "n,roll",
                "\"Rock 'n' roll\" -> APPLE",
            ])

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


if __name__ == "__main__":
    unittest.main()
