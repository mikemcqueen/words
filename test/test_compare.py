import json
import tempfile
import unittest
from pathlib import Path

import compare_native
from compare import (
    _aligned_chunk_generator,
    _block_generator_from_chunks,
    _parsed_eval_results_chunk_generator,
    _prefetch,
    eval_results_block_generator,
)


class EvalResultsGeneratorsTests(unittest.TestCase):
    def _write_result_file(self, directory, host, pairs):
        path = Path(directory) / f"pairs_third_p3_{host}.jsonl"
        with open(path, "w") as handle:
            for seq, pair in enumerate(pairs):
                json.dump({"pair": pair, "seq": seq, "logprobs": {" yes": -0.1, " no": -1.0}}, handle)
                handle.write("\n")
        return str(path)

    def test_parsed_chunk_generator_yields_aligned_chunks(self):
        pairs = [f"pair_{i}" for i in range(250)]
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_result_file(tmpdir, "hosta.tag1", pairs),
                self._write_result_file(tmpdir, "hostb.tag2", pairs),
            ]

            chunks = list(_parsed_eval_results_chunk_generator(files, chunk_size=100))

        self.assertEqual([100, 100, 50], [len(chunk["third.p3.tag1"]) for chunk in chunks])
        self.assertEqual(
            [row["pair"] for row in chunks[0]["third.p3.tag1"]],
            pairs[:100],
        )
        self.assertEqual(
            [row["pair"] for row in chunks[2]["third.p3.tag2"]],
            pairs[200:],
        )

    def test_parsed_chunk_generator_pair_mismatch_raises(self):
        primary_pairs = [f"pair_{i}" for i in range(120)]
        secondary_pairs = primary_pairs.copy()
        secondary_pairs[100] = "pair_mismatch"

        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_result_file(tmpdir, "hosta.tag1", primary_pairs),
                self._write_result_file(tmpdir, "hostb.tag2", secondary_pairs),
            ]

            with self.assertRaisesRegex(RuntimeError, "pair mismatch"):
                list(_parsed_eval_results_chunk_generator(files, chunk_size=100))

    def test_block_generator_batches_chunks_into_100_pair_blocks(self):
        pairs = [f"pair_{i}" for i in range(250)]
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_result_file(tmpdir, "hosta.tag1", pairs),
                self._write_result_file(tmpdir, "hostb.tag2", pairs),
            ]

            blocks = list(eval_results_block_generator(files))

        self.assertEqual([250], [len(next(iter(block.values()))) for block in blocks])
        self.assertEqual(
            set(blocks[0]["third.p3.tag1"].keys()),
            set(pairs),
        )

    def test_final_partial_block_is_yielded(self):
        pairs = [f"pair_{i}" for i in range(150)]
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_result_file(tmpdir, "hosta.tag1", pairs),
                self._write_result_file(tmpdir, "hostb.tag2", pairs),
            ]

            blocks = list(eval_results_block_generator(files))

        self.assertEqual(1, len(blocks))
        self.assertEqual(150, len(blocks[0]["third.p3.tag1"]))
        self.assertEqual(150, len(blocks[0]["third.p3.tag2"]))

    def test_prefetched_chunk_pipeline_matches_direct_blocks(self):
        pairs = [f"pair_{i}" for i in range(250)]
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_result_file(tmpdir, "hosta.tag1", pairs),
                self._write_result_file(tmpdir, "hostb.tag2", pairs),
            ]

            expected_blocks = list(eval_results_block_generator(files))
            prefetched_blocks = list(
                _block_generator_from_chunks(
                    _prefetch(_parsed_eval_results_chunk_generator(files, chunk_size=100)),
                    block_size=1000,
                )
            )

        self.assertEqual(expected_blocks, prefetched_blocks)

    def test_loader_auto_matches_python_blocks(self):
        pairs = [f"pair_{i}" for i in range(250)]
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_result_file(tmpdir, "hosta.tag1", pairs),
                self._write_result_file(tmpdir, "hostb.tag2", pairs),
            ]

            python_blocks = list(eval_results_block_generator(files, loader="python"))
            auto_blocks = list(eval_results_block_generator(files, loader="auto"))

        self.assertEqual(python_blocks, auto_blocks)

    def test_loader_native_requires_extension(self):
        if compare_native.native_available():
            self.skipTest("native extension is available")
        with self.assertRaisesRegex(RuntimeError, "native compare loader unavailable"):
            list(_aligned_chunk_generator(["a", "b"], loader="native"))

    def test_native_loader_matches_python_blocks_when_available(self):
        if not compare_native.native_available():
            self.skipTest("native extension is not available")

        pairs = [f"pair_{i}" for i in range(250)]
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_result_file(tmpdir, "hosta.tag1", pairs),
                self._write_result_file(tmpdir, "hostb.tag2", pairs),
            ]

            python_blocks = list(eval_results_block_generator(files, loader="python"))
            native_blocks = list(eval_results_block_generator(files, loader="native"))

        self.assertEqual(python_blocks, native_blocks)


if __name__ == "__main__":
    unittest.main()
