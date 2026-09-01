from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_splits import load_test_split, load_training_splits


class DataSplitTests(unittest.TestCase):
    def test_training_and_test_interfaces_keep_test_tokens_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            train_tokens = np.arange(10, dtype=np.uint16)
            validation_tokens = np.arange(100, 120, dtype=np.uint16)
            test_tokens_expected = np.arange(200, 208, dtype=np.uint16)
            train_tokens.tofile(data_dir / "train.bin")
            validation_tokens.tofile(data_dir / "val.bin")
            test_tokens_expected.tofile(data_dir / "test.bin")

            training_splits = load_training_splits(data_dir)
            test_tokens = load_test_split(data_dir)

            self.assertEqual(set(training_splits), {"train", "val"})
            np.testing.assert_array_equal(training_splits["train"], train_tokens)
            np.testing.assert_array_equal(
                training_splits["val"],
                validation_tokens,
            )
            np.testing.assert_array_equal(test_tokens, test_tokens_expected)

            del training_splits, test_tokens
            gc.collect()

    def test_missing_physical_test_file_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            np.arange(8, dtype=np.uint16).tofile(data_dir / "val.bin")

            with self.assertRaisesRegex(FileNotFoundError, "test.bin"):
                load_test_split(data_dir)


if __name__ == "__main__":
    unittest.main()
