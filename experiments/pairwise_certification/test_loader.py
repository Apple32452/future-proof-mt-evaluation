"""Contract tests for bounded development-only input loading."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from loader import EXPECTED_BASENAME, EXPECTED_SHA256, load_development_csv


class DevelopmentLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name)

    def make_csv(self, rows: list[list[object]], name: str = "development.csv") -> Path:
        path = self.directory / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
        return path

    def basic_rows(self) -> list[list[object]]:
        return [
            ["sentence_id", "source_text", "System-B", "System-A"],
            ["doc2_#_1", "quoted, text", 80, 50],
            ["doc1_#_0", "text", 40, 20],
            ["doc2_#_0", "text", 100, 0],
        ]

    def test_fixed_normalization_and_preserved_orders(self) -> None:
        path = self.make_csv(self.basic_rows())
        matrix, meta = load_development_csv(path)
        np.testing.assert_array_equal(matrix, [[0.8, 0.5], [0.4, 0.2], [1.0, 0.0]])
        self.assertEqual(meta["systems"], ["System-B", "System-A"])
        self.assertEqual(meta["items"], 3)
        self.assertEqual(meta["documents"], 2)
        self.assertEqual(meta["score_bounds_raw"], [0.0, 100.0])
        self.assertEqual(meta["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIsNone(meta["human_ratings_per_cell"])
        self.assertEqual(meta["data_role"], "development_only")

    def test_no_empirical_minmax_rescaling(self) -> None:
        rows = self.basic_rows()[:3]
        matrix, _ = load_development_csv(self.make_csv(rows))
        np.testing.assert_array_equal(matrix, [[0.8, 0.5], [0.4, 0.2]])

    def test_empty_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            load_development_csv(self.make_csv([]))

    def test_header_without_items(self) -> None:
        with self.assertRaisesRegex(ValueError, "no item rows"):
            load_development_csv(self.make_csv(self.basic_rows()[:1]))

    def test_missing_required_metadata(self) -> None:
        for column in ("sentence_id", "source_text"):
            rows = self.basic_rows()
            index = rows[0].index(column)
            for row in rows:
                del row[index]
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "requires"):
                load_development_csv(self.make_csv(rows))

    def test_missing_and_nonnumeric_scores(self) -> None:
        for value in ("", "not-a-score"):
            rows = self.basic_rows()
            rows[1][2] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonnumeric"):
                load_development_csv(self.make_csv(rows))

    def test_nonfinite_scores(self) -> None:
        for value in ("nan", "inf", "-inf"):
            rows = self.basic_rows()
            rows[1][2] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Nonfinite"):
                load_development_csv(self.make_csv(rows))

    def test_out_of_range_scores(self) -> None:
        for value in (-0.01, 100.01):
            rows = self.basic_rows()
            rows[1][2] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fixed bounds"):
                load_development_csv(self.make_csv(rows))

    def test_duplicate_ids(self) -> None:
        rows = self.basic_rows()
        rows[2][0] = rows[1][0]
        with self.assertRaisesRegex(ValueError, "Duplicate item"):
            load_development_csv(self.make_csv(rows))

    def test_empty_and_padded_ids(self) -> None:
        for value in ("", " ", " doc1"):
            rows = self.basic_rows()
            rows[1][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "item ID"):
                load_development_csv(self.make_csv(rows))

    def test_reference_detection(self) -> None:
        for name in ("refA", "RefA", "REFA"):
            rows = self.basic_rows()
            rows[0][2] = name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "reference"):
                load_development_csv(self.make_csv(rows))

    def test_duplicate_headers(self) -> None:
        rows = self.basic_rows()
        rows[0][2] = rows[0][3]
        with self.assertRaisesRegex(ValueError, "Duplicate column"):
            load_development_csv(self.make_csv(rows))

    def test_two_systems_required(self) -> None:
        rows = [row[:3] for row in self.basic_rows()]
        with self.assertRaisesRegex(ValueError, "two machine-system"):
            load_development_csv(self.make_csv(rows))

    def test_field_count_mismatch(self) -> None:
        rows = self.basic_rows()
        rows[1].pop()
        with self.assertRaisesRegex(ValueError, "number of fields"):
            load_development_csv(self.make_csv(rows))

    def test_historical_filename_requires_exact_checksum(self) -> None:
        path = self.make_csv(self.basic_rows(), EXPECTED_BASENAME)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_development_csv(path)

    def test_verified_hash_alone_enables_known_rating_cost(self) -> None:
        path = self.make_csv(self.basic_rows())
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        # This tests the trust gate without requiring the real dataset to be bundled.
        with patch("loader.EXPECTED_SHA256", actual_hash):
            _, meta = load_development_csv(path)
        self.assertEqual(meta["human_ratings_per_cell"], 2)

    def test_available_historical_matrix(self) -> None:
        path = Path(__file__).parent / "inputs" / EXPECTED_BASENAME
        if not path.is_file():
            self.skipTest("Optional development matrix not bundled; retrieve pinned upstream input")
        matrix, meta = load_development_csv(path)
        self.assertEqual(meta["sha256"], EXPECTED_SHA256)
        self.assertEqual(matrix.shape, (231, 20))
        self.assertEqual(meta["documents"], 128)
        self.assertEqual(meta["human_ratings_per_cell"], 2)
        self.assertTrue(np.isfinite(matrix).all())


if __name__ == "__main__":
    unittest.main()
