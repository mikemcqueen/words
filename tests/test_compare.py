import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import compare_native
from compare import (
    _block_generator_from_chunks,
    _parsed_eval_results_chunk_generator,
    _prefetch,
    eval_results_block_generator,
)
from score import score_eval_results


class EvalResultsGeneratorsTests(unittest.TestCase):
    def _write_result_file(self, directory, host, pairs):
        records = [
            {"pair": pair, "seq": seq, "logprobs": {" yes": -0.1, " no": -1.0}}
            for seq, pair in enumerate(pairs)
        ]
        return self._write_records_file(directory, host, records)

    def _write_records_file(self, directory, host, records):
        path = Path(directory) / f"pairs_third_p3_{host}.jsonl"
        with open(path, "w") as handle:
            for record in records:
                json.dump(record, handle)
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

    def test_eval_results_block_generator_matches_python_blocks(self):
        pairs = [f"pair_{i}" for i in range(250)]
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_result_file(tmpdir, "hosta.tag1", pairs),
                self._write_result_file(tmpdir, "hostb.tag2", pairs),
            ]

            expected_blocks = list(
                _block_generator_from_chunks(
                    _parsed_eval_results_chunk_generator(files, chunk_size=100),
                    block_size=1000,
                )
            )
            actual_blocks = list(eval_results_block_generator(files))

        self.assertEqual(expected_blocks, actual_blocks)

    def test_projected_loader_exposes_compact_arrays_when_available(self):
        if not compare_native.native_available():
            self.skipTest("native extension is not available")

        records_a = [
            {
                "pair": "pair_0",
                "seq": 0,
                "logprobs": {
                    "fwd": [{"YES": 0.9}, {"NO": 0.1}],
                    "rvs": [{"NO": 0.8}, {"YES": 0.2}],
                },
            },
            {
                "pair": "pair_1",
                "seq": 1,
                "logprobs": {
                    "fwd": [{"NO": 0.7}, {"YES": 0.3}],
                    "rvs": [{"YES": 0.6}, {"NO": 0.4}],
                },
            },
        ]
        records_b = [
            {
                "pair": "pair_0",
                "seq": 0,
                "logprobs": {
                    "fwd": [{"NO": 0.55}, {"YES": 0.45}],
                    "rvs": [{"NO": 0.51}, {"YES": 0.49}],
                },
            },
            {
                "pair": "pair_1",
                "seq": 1,
                "logprobs": {
                    "fwd": [{"YES": 0.65}, {"NO": 0.35}],
                    "rvs": [{"NO": 0.75}, {"YES": 0.25}],
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_records_file(tmpdir, "hosta.tag1", records_a),
                self._write_records_file(tmpdir, "hostb.tag2", records_b),
            ]

            block = next(compare_native.iter_projected_blocks(files, chunk_size=10))

        self.assertEqual(["third.p3.tag1", "third.p3.tag2"], list(block.keys()))
        self.assertEqual(["fwd", "rvs"], list(block.directions()))
        self.assertEqual(2, block.size)

        labels = np.asarray(block.labels())
        probs = np.asarray(block.probs())
        self.assertEqual((2, 2, 2), labels.shape)
        self.assertEqual((2, 2, 2), probs.shape)

        expected_labels = np.array(
            [
                [
                    [compare_native.LABEL_YES, compare_native.LABEL_NO],
                    [compare_native.LABEL_NO, compare_native.LABEL_YES],
                ],
                [
                    [compare_native.LABEL_NO, compare_native.LABEL_NO],
                    [compare_native.LABEL_YES, compare_native.LABEL_NO],
                ],
            ],
            dtype=np.uint8,
        )
        expected_probs = np.array(
            [
                [[0.9, 0.8], [0.7, 0.6]],
                [[0.55, 0.51], [0.65, 0.75]],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(expected_labels, labels)
        np.testing.assert_allclose(expected_probs, probs)

    def test_projected_loader_pair_mismatch_raises_when_available(self):
        if not compare_native.native_available():
            self.skipTest("native extension is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_result_file(tmpdir, "hosta.tag1", ["pair_0", "pair_1"]),
                self._write_result_file(tmpdir, "hostb.tag2", ["pair_0", "pair_x"]),
            ]

            with self.assertRaisesRegex(RuntimeError, "pair mismatch"):
                list(compare_native.iter_projected_blocks(files, chunk_size=10))

    def test_projected_loader_direction_mismatch_raises_when_available(self):
        if not compare_native.native_available():
            self.skipTest("native extension is not available")

        records_a = [
            {"pair": "pair_0", "seq": 0, "logprobs": {"fwd": [{"YES": 0.9}], "rvs": [{"NO": 0.1}]}}
        ]
        records_b = [
            {"pair": "pair_0", "seq": 0, "logprobs": {"fwd": [{"YES": 0.9}], "alt": [{"NO": 0.1}]}}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_records_file(tmpdir, "hosta.tag1", records_a),
                self._write_records_file(tmpdir, "hostb.tag2", records_b),
            ]

            with self.assertRaisesRegex(RuntimeError, "direction mismatch"):
                list(compare_native.iter_projected_blocks(files, chunk_size=10))

    def test_projected_labels_match_python_top_token_results_when_available(self):
        if not compare_native.native_available():
            self.skipTest("native extension is not available")

        records_a = [
            {
                "pair": "pair_0",
                "seq": 0,
                "logprobs": {"fwd": [{"YES": 0.9}, {"NO": 0.1}], "rvs": [{"NO": 0.8}, {"YES": 0.2}]},
            },
            {
                "pair": "pair_1",
                "seq": 1,
                "logprobs": {"fwd": [{"NO": 0.7}, {"YES": 0.3}], "rvs": [{"NO": 0.6}, {"YES": 0.4}]},
            },
        ]
        records_b = [
            {
                "pair": "pair_0",
                "seq": 0,
                "logprobs": {"fwd": [{"NO": 0.6}, {"YES": 0.4}], "rvs": [{"YES": 0.75}, {"NO": 0.25}]},
            },
            {
                "pair": "pair_1",
                "seq": 1,
                "logprobs": {"fwd": [{"YES": 0.55}, {"NO": 0.45}], "rvs": [{"NO": 0.7}, {"YES": 0.3}]},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                self._write_records_file(tmpdir, "hosta.tag1", records_a),
                self._write_records_file(tmpdir, "hostb.tag2", records_b),
            ]

            python_blocks = list(eval_results_block_generator(files))
            projected_block = next(compare_native.iter_projected_blocks(files, chunk_size=10))

        expected_yes = {}
        for key, rows in python_blocks[0].items():
            score_eval_results(rows, "top-token")
            expected_yes[key] = np.array([rows[pair]["yes"] for pair in rows], dtype=np.bool_)

        labels = np.asarray(projected_block.labels())
        actual_yes = np.any(labels == compare_native.LABEL_YES, axis=2)
        for file_index, key in enumerate(projected_block.keys()):
            np.testing.assert_array_equal(expected_yes[key], actual_yes[file_index])


if __name__ == "__main__":
    unittest.main()
