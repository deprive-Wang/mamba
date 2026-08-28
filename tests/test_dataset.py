"""token 流数据加载的边界与 label 错位测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataset import get_batch, load_token_stream, validate_token_range


class DatasetTests(unittest.TestCase):
    def test_memmap_batch_has_shifted_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.bin"
            np.arange(40, dtype=np.uint16).tofile(path)
            tokens = load_token_stream(path)
            try:
                generator = torch.Generator().manual_seed(1)
                x, y = get_batch(
                    tokens,
                    batch_size=4,
                    block_size=8,
                    generator=generator,
                )

                self.assertEqual(x.shape, (4, 8))
                self.assertEqual(x.dtype, torch.long)
                self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
                self.assertEqual(validate_token_range(tokens, 40), (0, 39))
            finally:
                # Windows 会锁住仍被 memmap 持有的临时文件，测试作为资源所有者显式关闭。
                tokens._mmap.close()

    def test_rejects_too_short_stream(self) -> None:
        tokens = np.arange(4, dtype=np.uint16)
        with self.assertRaises(ValueError):
            get_batch(tokens, batch_size=1, block_size=4)

    def test_rejects_out_of_vocab_token(self) -> None:
        tokens = np.array([0, 1, 7], dtype=np.uint16)
        with self.assertRaises(ValueError):
            validate_token_range(tokens, vocab_size=7)


if __name__ == "__main__":
    unittest.main()
