"""Offline CPU regressions for training, alignment and tiled/distributed examples."""
from __future__ import annotations

import math
import random
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from code.alignment.grpo import (
    GRPOTrainer, _Rollout, _generate_rollouts, _response_logprobs,
    compute_group_advantages, generate_solutions, grpo_loss,
)
from code.alignment.sft import SFTTrainer, _collate, compute_sft_loss, create_sft_dataset
from code.systems.ddp import SimpleDDP, cleanup_distributed, ddp_train_step
from code.systems.flash_attention import benchmark_flash_vs_standard, flash_attention_forward, standard_attention
from code.training.scheduler import CosineWithWarmup
from code.training.trainer import TextDataset, Trainer


class TinyLM(nn.Module):
    """A differentiable five-token table, with no pretrained downloads."""
    def __init__(self, raw_logits=False):
        super().__init__()
        self.table = nn.Parameter(torch.arange(25, dtype=torch.float32).reshape(5, 5) / 10)
        self.raw_logits = raw_logits
        self.modes, self.configs, self.generation_count = [], [], 0

    def forward(self, x=None, input_ids=None, attention_mask=None):
        self.modes.append(self.training)
        ids = input_ids if input_ids is not None else x
        logits = self.table[ids]
        return logits if self.raw_logits else SimpleNamespace(logits=logits)

    def generate(self, input_ids, generation_config, **kwargs):
        self.configs.append(generation_config)
        self.generation_count += 1
        token = 2 if self.generation_count % 2 else 3
        return torch.cat([input_ids, torch.tensor([[token, 0]], device=input_ids.device)], dim=1)


class BoundaryTokenizer:
    pad_token_id, eos_token_id, bos_token_id = 0, 0, 4

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        ids = {"a": [1], "b": [2], "c": [3], "ab": [3], "": []}[text]
        if add_special_tokens:
            ids = [4] + ids
        if return_tensors:
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join({0: "", 1: "a", 2: "b", 3: "c", 4: ""}[int(i)] for i in ids)


class TrainerTests(unittest.TestCase):
    def make_trainer(self, labels, batch_size=1):
        model = TinyLM(raw_logits=True)
        x = torch.zeros_like(labels)
        loader = DataLoader(TensorDataset(x, labels), batch_size=batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.5)
        scheduler = CosineWithWarmup(0.0, 0.0, 0, 10)
        return Trainer(model, optimizer, scheduler, loader, val_loader=loader, log_interval=100)

    def test_metrics_are_token_weighted_and_evaluate_restores_mode(self):
        labels = torch.tensor([[0, -100], [4, 4]])
        trainer = self.make_trainer(labels)
        expected = torch.nn.functional.cross_entropy(trainer.model(torch.zeros_like(labels)).reshape(-1, 5), labels.reshape(-1)).item()
        with patch("code.training.trainer.tqdm", None):
            self.assertAlmostEqual(trainer.evaluate()["loss"], expected, places=6)
            self.assertTrue(trainer.model.training)
            self.assertAlmostEqual(trainer.train_one_epoch()["loss"], expected, places=6)

    def test_all_ignored_batches_do_not_update_or_advance(self):
        trainer = self.make_trainer(torch.full((1, 2), -100))
        before = trainer.model.table.detach().clone()
        self.assertEqual(trainer._forward_loss(next(iter(trainer.train_loader))).item(), 0.0)
        with patch("code.training.trainer.tqdm", None):
            with self.assertRaises(ValueError):
                trainer.train_one_epoch()
            with self.assertRaises(ValueError):
                trainer.evaluate()
        self.assertEqual(trainer.global_step, 0)
        torch.testing.assert_close(trainer.model.table, before)

    def test_checkpoint_restores_rng_scheduler_and_loader_generator(self):
        trainer = self.make_trainer(torch.tensor([[0, 1]]))
        trainer.scheduler = CosineWithWarmup(0.1, 0.01, 2, 20)
        trainer.train_loader.generator = torch.Generator().manual_seed(12)
        trainer.global_step, trainer.epoch = 3, 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ckpt.pt"
            trainer.save_checkpoint(path, run="tiny")
            expected_torch, expected_python = torch.rand(3), random.random()
            expected_loader = torch.rand(3, generator=trainer.train_loader.generator)
            trainer.scheduler.max_steps, trainer.scheduler.max_lr = 1, 10
            torch.manual_seed(42)
            random.seed(42)
            trainer.train_loader.generator.manual_seed(42)
            self.assertEqual(trainer.load_checkpoint(path), {"run": "tiny"})
            torch.testing.assert_close(torch.rand(3), expected_torch)
            self.assertEqual(random.random(), expected_python)
            torch.testing.assert_close(torch.rand(3, generator=trainer.train_loader.generator), expected_loader)
        self.assertEqual(trainer.scheduler.max_steps, 20)
        self.assertEqual(trainer.scheduler.max_lr, 0.1)
        self.assertEqual((trainer.global_step, trainer.epoch), (3, 2))

    def test_text_dataset_shift_and_index_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.txt"
            path.write_text("0123456")
            dataset = TextDataset(path, 3, lambda text: [int(c) for c in text])
            torch.testing.assert_close(dataset[0][0], torch.tensor([0, 1, 2]))
            torch.testing.assert_close(dataset[0][1], torch.tensor([1, 2, 3]))
            torch.testing.assert_close(dataset[-1][0], torch.tensor([3, 4, 5]))
            with self.assertRaises(IndexError):
                dataset[2]


class SFTTests(unittest.TestCase):
    def sample(self, labels, mask):
        labels, mask = torch.tensor(labels), torch.tensor(mask, dtype=torch.float32)
        return {"input_ids": torch.zeros_like(labels), "attention_mask": torch.ones_like(labels), "labels": labels, "loss_mask": mask}

    def test_shift_mask_denominator_and_zero_loss_gradients(self):
        model = TinyLM()
        ids = torch.tensor([[0, 1, 2]])
        labels = torch.tensor([[-100, 2, -100]])
        mask = torch.ones_like(ids, dtype=torch.float32)
        actual = compute_sft_loss(model, ids, labels, mask)
        expected = torch.nn.functional.cross_entropy(model.table[0:1], torch.tensor([2]))
        torch.testing.assert_close(actual, expected)
        empty = compute_sft_loss(model, ids, torch.full_like(ids, -100), mask)
        self.assertEqual(empty.item(), 0.0)
        empty.backward()
        torch.testing.assert_close(model.table.grad, torch.zeros_like(model.table))

    def test_collator_uses_real_pad_id_and_padding_is_not_a_target(self):
        short, long = self.sample([-100, 2], [0, 1]), self.sample([-100, 1, 4], [0, 1, 1])
        batch = _collate([short, long], pad_token_id=4)
        self.assertEqual(batch["input_ids"][0, -1].item(), 4)
        self.assertEqual(batch["labels"][0, -1].item(), -100)
        self.assertEqual(batch["attention_mask"][0, -1].item(), 0)
        corrupted_labels = batch["labels"].clone()
        corrupted_labels[0, -1] = 999
        corrupted_mask = torch.ones_like(batch["loss_mask"])
        torch.testing.assert_close(
            compute_sft_loss(TinyLM(), batch["input_ids"], corrupted_labels, corrupted_mask, batch["attention_mask"]),
            compute_sft_loss(TinyLM(), batch["input_ids"], batch["labels"], batch["loss_mask"], batch["attention_mask"]),
        )

    def test_offset_tokens_crossing_role_boundary_are_excluded(self):
        conversation = [{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "bc"}]}]
        raw = create_sft_dataset(conversation)[0]
        start, end = raw["assistant_char_spans"][0]

        class OffsetTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [4, 1, 2, 3], "attention_mask": [1] * 4,
                        "offset_mapping": [(0, 0), (start - 1, start + 1), (start + 1, end), (end, end + 1)]}

        sample = create_sft_dataset(conversation, OffsetTokenizer())[0]
        self.assertEqual(sample["loss_mask"].tolist(), [0.0, 0.0, 1.0, 0.0])

    def test_eval_is_token_weighted_and_training_mode_survives_epochs(self):
        trainer = object.__new__(SFTTrainer)
        trainer.device, trainer.model = "cpu", TinyLM()
        samples = [self.sample([-100, 0], [0, 1]), self.sample([-100, 4, 4], [0, 1, 1])]
        loader = DataLoader(samples, batch_size=1, collate_fn=_collate)
        combined = _collate(samples)
        expected = compute_sft_loss(trainer.model, combined["input_ids"], combined["labels"], combined["loss_mask"], combined["attention_mask"]).item()
        self.assertAlmostEqual(trainer.evaluate(loader), expected, places=6)
        self.assertTrue(trainer.model.training)
        trainer.train(loader, loader, epochs=2, lr=0.0)
        # Each epoch has two training forwards followed by two evaluation forwards.
        self.assertEqual(trainer.model.modes[-8:], [True, True, False, False, True, True, False, False])


class GRPOTests(unittest.TestCase):
    def setUp(self):
        self.model, self.ref, self.tokenizer = TinyLM(), TinyLM(), BoundaryTokenizer()

    def test_group_centering_normalization_and_invalid_groups(self):
        rewards = torch.tensor([0., 1., 2., 2.], requires_grad=True)
        adv = compute_group_advantages(rewards, 2)
        torch.testing.assert_close(adv, torch.tensor([-1., 1., 0., 0.]))
        self.assertFalse(adv.requires_grad)
        torch.testing.assert_close(compute_group_advantages(rewards, 2, normalize_std=False), torch.tensor([-.5, .5, 0., 0.]))
        for group in (0, 1, 3):
            with self.assertRaises(ValueError):
                compute_group_advantages(rewards, group)

    def test_rollout_keeps_eos_tokens_old_probs_and_unwarped_sampling(self):
        rollouts = _generate_rollouts(self.model, self.tokenizer, ["a"], 2, 2, 1.0, 1.0, "cpu", record_logprobs=True)
        self.assertEqual(rollouts[0].input_ids.tolist(), [4, 1, 2, 0])
        self.assertEqual(rollouts[0].old_logprobs.numel(), 2)
        self.assertFalse(rollouts[0].old_logprobs.requires_grad)
        self.assertTrue(self.model.training)
        self.assertEqual(self.model.configs[0].pad_token_id, 0)
        self.assertEqual((self.model.configs[0].temperature, self.model.configs[0].top_p, self.model.configs[0].top_k), (1., 1., 0))
        self.assertEqual(generate_solutions(self.model, self.tokenizer, ["a"], 2), ["b", "c"])
        self.assertTrue(self.model.training)

    def test_clipping_for_both_advantage_signs_and_frozen_old_ref(self):
        for advantage, ratio, clipped in ((1., 2., True), (-1., .5, True), (1., .5, False), (-1., 2., False)):
            with self.subTest(advantage=advantage, ratio=ratio):
                self.model.zero_grad(set_to_none=True)
                ids = torch.tensor([4, 1, 2])
                lp = _response_logprobs(self.model, ids, 2).detach()
                old = (lp - math.log(ratio)).requires_grad_()
                adv = torch.tensor([advantage], requires_grad=True)
                loss, kl, pg = grpo_loss(self.model, self.ref, self.tokenizer, ["a"], ["b"], adv, 0., old_logprobs=[old])
                expected_ratio = min(ratio, 1.2) if advantage > 0 else max(ratio, .8)
                self.assertAlmostEqual(loss.item(), -advantage * expected_ratio, places=6)
                loss.backward()
                self.assertEqual(bool(self.model.table.grad.abs().sum() == 0), clipped)
                self.assertIsNone(old.grad)
                self.assertIsNone(adv.grad)
                self.assertIsNone(self.ref.table.grad)
                self.assertEqual(kl.item(), 0.)

    def test_kl_estimator_is_nonnegative_and_ref_has_no_gradient(self):
        with torch.no_grad():
            self.ref.table[1, 2] += 1
        ids = torch.tensor([4, 1, 2])
        old = _response_logprobs(self.model, ids, 2).detach()
        loss, kl, _ = grpo_loss(self.model, self.ref, self.tokenizer, ["a"], ["b"], torch.tensor([0.]), .1, old_logprobs=[old])
        self.assertGreater(kl.item(), 0.)
        loss.backward()
        self.assertGreater(self.model.table.grad.abs().sum().item(), 0.)
        self.assertIsNone(self.ref.table.grad)

    def test_zero_advantage_extreme_ratio_and_missing_old_policy(self):
        old = torch.tensor([-1000.], requires_grad=True)
        loss, _, _ = grpo_loss(self.model, self.ref, self.tokenizer, ["a"], ["b"], torch.tensor([0.]), 0., old_logprobs=[old])
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.table.grad).all())
        with self.assertRaises(ValueError):
            grpo_loss(self.model, self.ref, self.tokenizer, ["a"], ["b"], torch.tensor([1.]), 0.)

    def test_train_step_runs_offline_with_exact_generated_tokens(self):
        trainer = object.__new__(GRPOTrainer)
        trainer.device, trainer.model, trainer.ref_model, trainer.tokenizer = "cpu", self.model, self.ref, self.tokenizer
        trainer.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        before = self.model.table.detach().clone()
        metrics = trainer.train_step(["a"], ["b"], 2, max_new_tokens=2)
        self.assertEqual(metrics["reward_mean"], .5)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertGreater((self.model.table.detach() - before).abs().sum().item(), 0.)
        self.assertIsNone(self.ref.table.grad)


def _distributed_regression(rank, rendezvous):
    """Real local gloo collective; each process raises on incorrect synchronization."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=30))
    try:
        bare = nn.Linear(1, 1, bias=False)
        bare.weight.data.fill_(rank + 1.)
        wrapped = SimpleDDP(bare)
        torch.testing.assert_close(bare.weight, torch.tensor([[1.]]))
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=.1)
        ddp_train_step(wrapped, torch.tensor([[rank + 1.]]), torch.zeros((1, 1)), nn.MSELoss(), optimizer)
        # Local gradients are 2 and 8, their mean is 5, so both replicas become .5.
        torch.testing.assert_close(bare.weight, torch.tensor([[.5]]))
        wrapped.close()
        # Every rank must call new_group in the same order, even non-members.
        singleton = dist.new_group(ranks=[0])
        other = dist.new_group(ranks=[1])
        singleton = singleton if rank == 0 else other
        single_bare = nn.Linear(1, 1, bias=False)
        single_bare.weight.data.fill_(1.)
        single = SimpleDDP(single_bare, process_group=singleton)
        self_grad = torch.tensor([[rank + 1.]])
        single(self_grad).sum().backward()
        torch.testing.assert_close(single_bare.weight.grad, self_grad)
        single.close()
    finally:
        cleanup_distributed()


class SystemsTests(unittest.TestCase):
    def test_tiled_attention_matches_noncontiguous_inputs_and_gradients(self):
        torch.manual_seed(8)
        tensors = [torch.randn(2, 3, 7, 4, dtype=torch.float64).transpose(0, 1).requires_grad_() for _ in range(3)]
        tiled = flash_attention_forward(*tensors, block_size=3)
        standard = standard_attention(*tensors)
        torch.testing.assert_close(tiled, standard, rtol=1e-10, atol=1e-10)
        grads_tiled = torch.autograd.grad(tiled.square().sum(), tensors)
        grads_standard = torch.autograd.grad(standard.square().sum(), tensors)
        for actual, expected in zip(grads_tiled, grads_standard):
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)

    def test_half_precision_uses_stable_accumulation_and_invalid_blocks_fail(self):
        q, k, v = [torch.randn(1, 11, 4).to(torch.bfloat16) for _ in range(3)]
        torch.testing.assert_close(flash_attention_forward(q, k, v, 3), standard_attention(q, k, v), rtol=.008, atol=.004)
        for block in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                flash_attention_forward(q, k, v, block)
        with self.assertRaises(ValueError):
            benchmark_flash_vs_standard(n_iters=0)
        with self.assertRaises(ValueError):
            benchmark_flash_vs_standard(device="mps")
        with self.assertRaises(ValueError):
            standard_attention(q.long(), k.long(), v.long())

    def test_benchmark_excludes_warmup_from_timing(self):
        with patch("code.systems.flash_attention.time.perf_counter", side_effect=[1., 3., 10., 14.]):
            standard, tiled = benchmark_flash_vs_standard(seq_len=2, dim=2, batch=1, n_warmup=3, n_iters=2)
        self.assertEqual(standard["time_ms"], 1000.)
        self.assertEqual(tiled["time_ms"], 2000.)
        self.assertIsNone(standard["peak_mem_bytes"])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "gloo unavailable")
    def test_two_process_gloo_initial_broadcast_gradient_average_and_subgroups(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_regression, args=(str(Path(directory) / "store"),), nprocs=2, join=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
