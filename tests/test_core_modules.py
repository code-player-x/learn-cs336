"""CPU regressions for the byte BPE, model, and hand-written optimizer."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from code.model.transformer import (
    MultiHeadAttention,
    RMSNorm,
    RotaryEmbedding,
    TransformerLM,
    apply_rotary_emb,
)
from code.tokenizer.bpe import BPETokenizer, merge
from code.training.optimizer import AdamW
from code.training.scheduler import CosineWithWarmup


class BPETests(unittest.TestCase):
    def test_byte_tuple_ties_not_allocated_ids(self):
        tokenizer = BPETokenizer(259)
        tokenizer.train("aaaa zzzz")
        self.assertEqual(tokenizer.merges, [(122, 122), (97, 97), (256, 256)])

    def test_unicode_round_trip_and_saved_rules(self):
        text = "hello! 你好，世界 café 👋\n\t123\n"
        tokenizer = BPETokenizer(280)
        tokenizer.train(text * 2)
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(tokenizer.encode(""), [])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "tokenizer.json")
            tokenizer.save(path)
            restored = BPETokenizer.from_file(path)
        self.assertEqual(restored.encode(text), ids)
        self.assertEqual(restored.vocab, tokenizer.vocab)

    def test_base_vocab_retraining_persists_target(self):
        tokenizer = BPETokenizer(280)
        tokenizer.train("aaaa", vocab_size=256)
        tokenizer.train("aaaa")
        self.assertEqual(tokenizer.vocab_size, 256)

    def test_non_overlapping_merge_and_invalid_size(self):
        self.assertEqual(merge([1, 1, 1, 1, 1], (1, 1), 256), [256, 256, 1])
        for size in (255, 256.5):
            with self.assertRaises(ValueError):
                BPETokenizer(size)


class OptimizerTests(unittest.TestCase):
    def test_matches_pytorch_with_decay_groups_and_no_grad(self):
        torch.manual_seed(7)
        params = [torch.nn.Parameter(torch.randn(3, dtype=torch.float64)) for _ in range(3)]
        refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
        groups = [{"params": params[:2], "lr": 0.1, "weight_decay": 0.5}, {"params": params[2:], "lr": 0.03}]
        ref_groups = [{"params": refs[:2], "lr": 0.1, "weight_decay": 0.5}, {"params": refs[2:], "lr": 0.03}]
        ours = AdamW(iter(groups), betas=(0.8, 0.95), weight_decay=0.0)
        reference = torch.optim.AdamW(ref_groups, betas=(0.8, 0.95), weight_decay=0.0)
        for step in range(8):
            for index, (p, r) in enumerate(zip(params, refs)):
                gradient = torch.randn_like(p) if index != 1 or step % 2 == 0 else None
                p.grad = gradient
                r.grad = None if gradient is None else gradient.clone()
            ours.step()
            reference.step()
            for p, r in zip(params, refs):
                torch.testing.assert_close(p, r, rtol=1e-12, atol=1e-12)

    def test_checkpoint_restores_hyperparameters_and_has_no_aliases(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = AdamW([parameter], lr=0.03, betas=(0.7, 0.9), eps=1e-6, weight_decay=0.2)
        parameter.grad = torch.tensor([0.2, -0.4])
        optimizer.step()
        saved_parameter = parameter.detach().clone()
        checkpoint = optimizer.state_dict()
        saved_moment = checkpoint["state"][0]["exp_avg"].clone()
        restored_parameter = torch.nn.Parameter(saved_parameter.clone())
        restored = AdamW([restored_parameter], lr=1.0, weight_decay=0.0)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.param_groups[0]["lr"], 0.03)
        self.assertEqual(restored.param_groups[0]["betas"], (0.7, 0.9))
        self.assertEqual(checkpoint["param_groups"][0]["params"], [0])
        # Neither further training nor mutation of a returned checkpoint can
        # mutate another optimizer's state or a previously captured snapshot.
        optimizer.step()
        torch.testing.assert_close(checkpoint["state"][0]["exp_avg"], saved_moment)
        checkpoint["state"][0]["exp_avg"].fill_(100)
        torch.testing.assert_close(restored.state[restored_parameter]["exp_avg"], saved_moment)
        parameter.data.copy_(saved_parameter)
        optimizer.load_state_dict(restored.state_dict())
        restored_parameter.grad = parameter.grad.clone()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(parameter, restored_parameter)

    def test_invalid_optimizer_hyperparameters_and_group_layout(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        for kwargs in ({"lr": -1}, {"eps": -1}, {"weight_decay": -1}, {"betas": (1.0, 0.9)}, {"lr": float("nan")}):
            with self.assertRaises(ValueError):
                AdamW([parameter], **kwargs)
        with self.assertRaises(ValueError):
            AdamW([parameter, parameter])
        optimizer = AdamW([parameter])
        saved = optimizer.state_dict()
        other = AdamW([{"params": [parameter]}, {"params": []}])
        with self.assertRaises(ValueError):
            other.load_state_dict(saved)

    def test_scheduler_boundaries_and_invalid_configuration(self):
        scheduler = CosineWithWarmup(1.0, 0.1, 2, 10)
        self.assertEqual(scheduler.get_lr(0), 0.5)
        self.assertEqual(scheduler.get_lr(1), 1.0)
        self.assertEqual(scheduler.get_lr(2), 1.0)
        self.assertEqual(scheduler.get_lr(10), 0.1)
        self.assertEqual(scheduler.get_lr(20), 0.1)
        values = [scheduler.get_lr(step) for step in range(2, 11)]
        self.assertEqual(values, sorted(values, reverse=True))
        all_warmup = CosineWithWarmup(1.0, 0.1, 2, 2)
        self.assertEqual(all_warmup.get_lr(2), 0.1)
        for args in ((1, 0, 3, 2), (0.1, 1, 0, 2), (math.inf, 0, 0, 2), (1, 0, 0.5, 2)):
            with self.assertRaises(ValueError):
                CosineWithWarmup(*args)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_head_registered_and_tied_before_forward_checkpoint(self):
        model = TransformerLM(32, 16, 1, 2)
        before = model.state_dict()
        self.assertIn("lm_head.weight", before)
        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)
        logits = model(torch.tensor([[1, 2, 3]]))
        self.assertEqual(set(model.state_dict()), set(before))
        restored = TransformerLM(32, 16, 1, 2)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(torch.tensor([[1, 2, 3]])), logits)

    def test_left_padding_and_empty_rows_are_finite_with_finite_gradients(self):
        model = TransformerLM(32, 16, 1, 2, pad_token_id=0)
        ids = torch.tensor([[0, 0, 5, 6], [0, 0, 0, 0]])
        logits = model(ids)
        self.assertTrue(torch.isfinite(logits).all())
        unpadded = model(torch.tensor([[5, 6]]))
        torch.testing.assert_close(logits[:1, 2:], unpadded, atol=1e-6, rtol=1e-5)
        logits.square().mean().backward()
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_causal_and_grouped_attention(self):
        model = TransformerLM(32, 16, 1, 4, n_kv_heads=2)
        original = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 8, 9]])
        torch.testing.assert_close(model(original)[:, :2], model(changed)[:, :2])
        for dims in ((15, 2, None), (12, 4, None), (16, 4, 3), (16, 0, None), (16, 2, 0)):
            with self.assertRaises(ValueError):
                MultiHeadAttention(dims[0], dims[1], n_kv_heads=dims[2])

    def test_rmsnorm_precision_and_output_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            norm = RMSNorm(8)
            if dtype == torch.float64:
                norm = norm.double()
            x = torch.randn(2, 8).to(dtype)
            out = norm(x)
            self.assertEqual(out.dtype, dtype)
            expected = (x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + norm.eps)).to(dtype)
            torch.testing.assert_close(out, expected, rtol=0.01 if dtype == torch.bfloat16 else 1e-3, atol=1e-3)

    def test_rope_half_model_retains_long_position_precision(self):
        rope = RotaryEmbedding(8, base=10000.0)
        x = torch.ones(1, 4096, 8, dtype=torch.float16)
        expected_cos, expected_sin = rope(x)
        rope.half()
        cos, sin = rope(x)
        torch.testing.assert_close(cos, expected_cos, rtol=0, atol=0)
        torch.testing.assert_close(sin, expected_sin, rtol=0, atol=0)
        q = torch.randn(1, 2, 4, 8)
        cos, sin = rope(q.transpose(1, 2), seq_len=4)
        rotated, _ = apply_rotary_emb(q, q, cos[None, None], sin[None, None])
        torch.testing.assert_close(rotated.norm(dim=-1), q.norm(dim=-1))

    def test_half_attention_large_dot_products_are_finite(self):
        attention = MultiHeadAttention(8, 2).half()
        x = torch.full((1, 3, 8), 500.0, dtype=torch.float16)
        self.assertTrue(torch.isfinite(attention(x)).all())

    def test_generation_tracks_individual_eos_and_restores_train_mode(self):
        model = TransformerLM(4, 8, 0, 2)
        calls = []

        def forward(ids):
            index = len(calls)
            calls.append(ids)
            logits = torch.full((2, ids.shape[1], 4), -100.0)
            logits[0, -1, 3 if index == 0 else 1] = 0.0
            logits[1, -1, 3 if index == 2 else 2] = 0.0
            return logits

        with patch.object(model, "forward", side_effect=forward):
            out = model.generate(torch.tensor([[1], [1]]), 5, temperature=0, eos_token_id=3)
        self.assertEqual(out.tolist(), [[1, 3, 3, 3], [1, 2, 2, 3]])
        self.assertTrue(model.training)
        with patch.object(model, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                model.generate(torch.tensor([[1]]), 1)
        self.assertTrue(model.training)

    def test_nucleus_keeps_first_threshold_crossing_token(self):
        model = TransformerLM(3, 8, 0, 2)
        logits = torch.tensor([[[0.6, 0.3, 0.1]]]).log()
        captured = []

        def sample(probabilities, num_samples):
            captured.append(probabilities.clone())
            return torch.tensor([[1]])

        with patch.object(model, "forward", return_value=logits), patch("torch.multinomial", side_effect=sample):
            out = model.generate(torch.tensor([[0]]), 1, top_p=0.7)
        self.assertEqual(out.tolist(), [[0, 1]])
        torch.testing.assert_close(captured[0], torch.tensor([[2 / 3, 1 / 3, 0.0]]))

    def test_generation_validates_padding_sampling_and_prompt(self):
        model = TransformerLM(4, 8, 0, 2, pad_token_id=0)
        for kwargs in ({"temperature": -1}, {"top_p": 0}, {"top_p": 1.1}, {"max_new_tokens": -1}):
            options = {"max_new_tokens": 1, **kwargs}
            with self.assertRaises(ValueError):
                model.generate(torch.tensor([[1]]), **options)
        for prompt in (torch.tensor([[1, 0]]), torch.empty((1, 0), dtype=torch.long)):
            with self.assertRaises(ValueError):
                model.generate(prompt, 1)

    def test_tiny_positive_temperature_does_not_produce_nan(self):
        model = TransformerLM(3, 8, 0, 2)
        with patch.object(model, "forward", return_value=torch.tensor([[[1.0, 2.0, 3.0]]])):
            generated = model.generate(torch.tensor([[0]]), 1, temperature=1e-300)
        self.assertEqual(generated.tolist(), [[0, 2]])


if __name__ == "__main__":
    unittest.main()
