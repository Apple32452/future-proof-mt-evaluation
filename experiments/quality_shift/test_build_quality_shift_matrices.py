"""Focused tests for strict WMT25 human-score ingestion."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from build_quality_shift_matrices import check_pilot_matrix, read_matrices


class MatrixBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.data = self.directory / "human.jsonl"
        self.rows = [
            {
                "doc_id": f"toy_#_news_#_doc{doc}_#_0",
                "src_text": f"source {doc}",
                "scores": {
                    **{f"sys_{i:02d}": [{"score": 10.0 + i}, {"score": 30.0 + i},
                                        {"score": None}]
                       for i in reversed(range(12))},
                    "refA": [{"score": 100}],
                },
            }
            for doc in [2, 1]
        ]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_rows(self, rows=None) -> None:
        self.data.write_text("".join(json.dumps(r) + "\n" for r in
                                     (self.rows if rows is None else rows)), encoding="utf-8")

    def test_order_means_and_reference_exclusion(self) -> None:
        self.write_rows()
        matrices, audit = read_matrices(self.data, ("toy",))
        frame = matrices["toy"]
        self.assertEqual(frame.sentence_id.tolist(), [r["doc_id"] for r in self.rows])
        self.assertEqual(list(frame)[2:], [f"sys_{i:02d}" for i in range(12)])
        self.assertNotIn("refA", frame)
        np.testing.assert_array_equal(frame.sys_00.to_numpy(), [20.0, 20.0])
        self.assertEqual(audit[0]["n_null_annotations_skipped"], 24)
        self.assertEqual(audit[0]["n_annotations_used"], 48)
        self.assertEqual(audit[0]["n_documents"], 2)

    def test_missing_cell_is_error_not_filtered(self) -> None:
        del self.rows[0]["scores"]["sys_03"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "1 missing score cells"):
            read_matrices(self.data, ("toy",))

    def test_all_null_cell_is_error(self) -> None:
        self.rows[0]["scores"]["sys_03"] = [{"score": None}]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "1 missing score cells"):
            read_matrices(self.data, ("toy",))

    def test_duplicate_id_is_error(self) -> None:
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate item ID"):
            read_matrices(self.data, ("toy",))

    def test_nonfinite_score_is_error(self) -> None:
        self.rows[0]["scores"]["sys_03"][0]["score"] = float("nan")
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Non-finite score"):
            read_matrices(self.data, ("toy",))

    def test_bad_document_schema_is_error(self) -> None:
        self.rows[0]["doc_id"] = "toy_#_doc1"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Unexpected WMT item-ID schema"):
            read_matrices(self.data, ("toy",))

    def test_reference_check_covers_values_and_order(self) -> None:
        self.write_rows()
        matrices, _ = read_matrices(self.data, ("toy",))
        frame = matrices["toy"]
        reference = self.directory / "reference.csv"
        frame.to_csv(reference, index=False)
        self.assertEqual(check_pilot_matrix(frame, reference)["status"], "exact data match")
        with self.assertRaisesRegex(ValueError, "sentence_id"):
            check_pilot_matrix(frame.iloc[::-1], reference)
        changed = frame.copy()
        changed.loc[0, "sys_00"] += 1
        with self.assertRaisesRegex(ValueError, "scores differ"):
            check_pilot_matrix(changed, reference)

    def test_duplicate_language_request_is_error(self) -> None:
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "without duplicates"):
            read_matrices(self.data, ("toy", "toy"))


if __name__ == "__main__":
    unittest.main()
