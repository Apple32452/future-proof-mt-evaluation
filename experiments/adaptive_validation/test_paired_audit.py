"""Unit tests for paired randomization analysis; no human matrices needed."""
import tempfile
import unittest
from pathlib import Path
from paired_audit import paired_interval, evaluate


class TestPairedAudit(unittest.TestCase):
    def test_constant_relative_saving(self):
        result = paired_interval(
            [100.0] * 10, [90.0] * 10,
            mode="relative_saving_percent", resamples=100, seed=1)
        self.assertAlmostEqual(result["estimate"], 10.0)
        self.assertAlmostEqual(result["percentile_95_interval"][0], 10.0)
        self.assertAlmostEqual(result["percentile_95_interval"][1], 10.0)

    def test_posthoc_top1_sign(self):
        result = paired_interval(
            [0, 1] * 5, [1] * 10,
            mode="difference", resamples=100, seed=7)
        self.assertAlmostEqual(result["estimate"], .5)

    def test_mismatched_replicate_count_rejected(self):
        with self.assertRaises(ValueError):
            paired_interval([1.0], [1.0, 2.0],
                            mode="absolute_saving_cells", resamples=100, seed=7)

    def test_missing_trial_files_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                evaluate(Path(temp), resamples=100)


if __name__ == "__main__":
    unittest.main()
