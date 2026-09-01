from __future__ import annotations

import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "当前 Python 环境没有 PyTorch")
class EvaluateTests(unittest.TestCase):
    def test_tensorboard_defaults_to_the_project_log_directory(self) -> None:
        from evaluate import parse_args as parse_evaluation_args
        from train import parse_args as parse_training_args

        expected_dir = Path(__file__).resolve().parents[1] / "tf-logs"
        with patch.object(sys, "argv", ["evaluate.py"]):
            evaluation_args = parse_evaluation_args()
        with patch.object(sys, "argv", ["train.py"]):
            training_args = parse_training_args()

        self.assertEqual(evaluation_args.tensorboard_dir, expected_dir)
        self.assertEqual(training_args.tensorboard_dir, expected_dir)

    def test_best_checkpoint_is_evaluated_on_reserved_test_tokens(self) -> None:
        from evaluate import evaluate_checkpoint, write_evaluation_summary
        from model import MambaConfig, MambaLanguageModel
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        assert torch is not None
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            data_dir.mkdir()
            test_tokens = (np.arange(16) % 16).astype(np.uint16)
            test_tokens.tofile(data_dir / "test.bin")

            config = MambaConfig(
                vocab_size=16,
                block_size=4,
                d_model=8,
                n_layers=1,
                d_state=2,
                d_conv=2,
                expand=1,
            )
            model = MambaLanguageModel(config)
            checkpoint_path = root / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": 7,
                    "best_val_loss": 2.5,
                    "config": asdict(config),
                },
                checkpoint_path,
            )

            result = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                batch_size=2,
                eval_batches=2,
                seed=123,
                device=torch.device("cpu"),
            )

            self.assertEqual(result["checkpoint_step"], 7)
            self.assertEqual(result["test_tokens_total"], 16)
            self.assertEqual(result["evaluated_tokens"], 16)
            self.assertTrue(math.isfinite(float(result["test_loss"])))
            self.assertTrue(math.isfinite(float(result["test_perplexity"])))

            log_dir = root / "tf-logs" / "mamba-formal-1000"
            write_evaluation_summary(result, log_dir)
            accumulator = EventAccumulator(str(log_dir))
            accumulator.Reload()
            scalar_tags = accumulator.Tags()["scalars"]
            self.assertIn("test/loss", scalar_tags)
            test_loss_event = accumulator.Scalars("test/loss")[0]
            self.assertEqual(test_loss_event.step, 7)
            self.assertAlmostEqual(
                test_loss_event.value,
                float(result["test_loss"]),
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
