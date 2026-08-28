"""Mamba 模型的外部行为回归测试。"""

import unittest

import torch

from model import MambaConfig, MambaLanguageModel, MambaMixer, ResidualMambaBlock


class MambaModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = MambaConfig(
            vocab_size=31,
            block_size=12,
            d_model=16,
            n_layers=2,
            d_state=4,
            d_conv=3,
            expand=2,
        )

    def test_mixer_and_residual_preserve_shape(self) -> None:
        hidden_states = torch.randn(2, 12, self.config.d_model)
        mixer = MambaMixer(self.config)
        block = ResidualMambaBlock(self.config)

        self.assertEqual(mixer(hidden_states).shape, hidden_states.shape)
        self.assertEqual(block(hidden_states).shape, hidden_states.shape)

    def test_mixer_is_causal(self) -> None:
        mixer = MambaMixer(self.config).eval()
        hidden_states = torch.randn(2, 12, self.config.d_model)
        changed = hidden_states.clone()
        changed[:, -1] += 3.0

        with torch.no_grad():
            output = mixer(hidden_states)
            changed_output = mixer(changed)

        self.assertTrue(
            torch.allclose(
                output[:, :-1],
                changed_output[:, :-1],
                atol=1e-5,
                rtol=1e-4,
            )
        )
        self.assertFalse(torch.allclose(output[:, -1], changed_output[:, -1]))

    def test_language_model_forward_and_backward(self) -> None:
        model = MambaLanguageModel(self.config)
        x = torch.randint(0, self.config.vocab_size, (2, 12))
        y = torch.randint(0, self.config.vocab_size, (2, 12))

        logits, loss = model(x, y)

        self.assertEqual(logits.shape, (2, 12, self.config.vocab_size))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.layers[0].mixer.A_log.grad)

    def test_config_resolves_official_auto_dt_rank(self) -> None:
        self.assertEqual(MambaConfig(d_model=128).resolved_dt_rank, 8)
        self.assertEqual(MambaConfig(d_model=129).resolved_dt_rank, 9)


if __name__ == "__main__":
    unittest.main()
