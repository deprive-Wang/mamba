"""训练状态持久化的回归测试。"""

import tempfile
import unittest
from pathlib import Path

import torch

from model import MambaConfig, MambaLanguageModel
from train import configure_optimizer, load_checkpoint, save_checkpoint


class TrainStateTests(unittest.TestCase):
    def test_checkpoint_round_trip(self) -> None:
        config = MambaConfig(
            vocab_size=17,
            block_size=4,
            d_model=8,
            n_layers=1,
            d_state=2,
            d_conv=2,
        )
        model = MambaLanguageModel(config)
        optimizer = configure_optimizer(model, 3e-4, torch.device("cpu"))
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        original_embedding = model.token_embedding.weight.detach().clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "state.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scaler,
                step=3,
                best_val_loss=1.25,
            )
            with torch.no_grad():
                model.token_embedding.weight.zero_()

            step, best_val_loss = load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scaler,
                torch.device("cpu"),
            )

        self.assertEqual(step, 3)
        self.assertEqual(best_val_loss, 1.25)
        self.assertTrue(
            torch.equal(model.token_embedding.weight, original_embedding)
        )


if __name__ == "__main__":
    unittest.main()
