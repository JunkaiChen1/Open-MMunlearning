import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trainer.unlearn.manu import (  # noqa: E402
    MANU,
    apply_output_neuron_masks,
    compute_combined_importance,
    compute_topk_masks,
)
from trainer.unlearn.mmunlearner import compute_saliency_mask  # noqa: E402
from trainer.unlearn.mip_editor import (  # noqa: E402
    apply_input_path_mask,
    compute_masked_representation_loss,
)
from trainer.unlearn.rmu import RMU  # noqa: E402
from trainer.utils import compute_kl_divergence, compute_undial_loss  # noqa: E402
from model import _apply_manu_masks, _sculpt_smfa_delta  # noqa: E402


class MultimodalUnlearningMethodTest(unittest.TestCase):
    def test_smfa_sculpt_depends_on_k(self):
        delta = torch.tensor([[2.0, -2.0]])
        retain = torch.tensor([[-1.0, 1.0]])

        aggressive = _sculpt_smfa_delta(delta, retain, k=0.5)
        conservative = _sculpt_smfa_delta(delta, retain, k=3.0)

        self.assertTrue(torch.equal(aggressive, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(conservative, delta))

    def test_manu_artifact_masks_exact_model_rows(self):
        model = nn.Sequential(nn.Linear(3, 4, bias=True))
        model[0].weight.data.fill_(1.0)
        model[0].bias.data.fill_(1.0)
        masks = {"0": torch.tensor([False, True, False, True])}

        applied = _apply_manu_masks(model, masks)

        self.assertEqual(applied, 2)
        self.assertTrue(torch.equal(model[0].weight[1], torch.zeros(3)))
        self.assertTrue(torch.equal(model[0].weight[3], torch.zeros(3)))
        self.assertEqual(model[0].bias[1].item(), 0.0)
        self.assertTrue(torch.equal(model[0].weight[0], torch.ones(3)))

    def test_mip_path_mask_edits_input_columns(self):
        linear = nn.Linear(4, 3, bias=True)
        linear.weight.data.fill_(1.0)
        linear.bias.data.fill_(1.0)

        apply_input_path_mask(
            linear,
            linear,
            torch.tensor([False, True, False, True]),
        )

        self.assertTrue(torch.equal(linear.weight[:, 1], torch.zeros(3)))
        self.assertTrue(torch.equal(linear.weight[:, 3], torch.zeros(3)))
        self.assertTrue(torch.equal(linear.weight[:, 0], torch.ones(3)))
        self.assertTrue(torch.equal(linear.bias, torch.ones(3)))

    def test_mip_representation_loss_ignores_non_answer_tokens(self):
        activations = torch.ones(2, 3, 4, requires_grad=True)
        target = torch.zeros(1, 1, 4)
        labels = torch.tensor(
            [
                [-100, 1, -100],
                [1, 2, -100],
            ]
        )

        loss = compute_masked_representation_loss(activations, target, labels)

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))
        loss.backward()
        self.assertTrue(torch.equal(activations.grad[0, 0], torch.zeros(4)))
        self.assertGreater(activations.grad[0, 1].abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(activations.grad[1, 2], torch.zeros(4)))

    def test_kl_divergence_uses_only_shifted_supervised_tokens(self):
        class FixedLogitModel(nn.Module):
            def __init__(self, logits, trainable=False):
                super().__init__()
                if trainable:
                    self.logits = nn.Parameter(logits.clone())
                else:
                    self.register_buffer("logits", logits.clone())

            def forward(self, **inputs):
                return SimpleNamespace(logits=self.logits)

        current_logits = torch.tensor(
            [[[1.0, 0.0, -1.0], [20.0, -20.0, 0.0], [0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]]
        )
        reference_logits = torch.tensor(
            [[[0.5, 1.0, -0.5], [-20.0, 20.0, 0.0], [1.5, -0.5, 0.5], [-4.0, -5.0, -6.0]]]
        )
        current = FixedLogitModel(current_logits, trainable=True)
        reference = FixedLogitModel(reference_logits)
        inputs = {
            "input_ids": torch.tensor([[4, 5, 6, 7]]),
            "labels": torch.tensor([[-100, 1, -100, 2]]),
        }

        loss, outputs = compute_kl_divergence(current, reference, inputs)

        expected = torch.nn.functional.kl_div(
            current_logits[0, [0, 2]].log_softmax(dim=-1),
            reference_logits[0, [0, 2]].log_softmax(dim=-1),
            reduction="batchmean",
            log_target=True,
        )
        self.assertTrue(torch.allclose(loss, expected))
        self.assertIs(outputs.logits, current.logits)

        loss.backward()
        self.assertGreater(current.logits.grad[0, [0, 2]].abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(current.logits.grad[0, 1], torch.zeros(3)))
        self.assertTrue(torch.equal(current.logits.grad[0, 3], torch.zeros(3)))
        self.assertIsNone(reference.logits.grad)

    def test_kl_divergence_rejects_batches_without_supervised_tokens(self):
        class FixedLogitModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(1, 3, 4))

            def forward(self, **inputs):
                return SimpleNamespace(logits=self.logits)

        inputs = {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.full((1, 3), -100),
        }
        with self.assertRaisesRegex(ValueError, "at least one supervised token"):
            compute_kl_divergence(FixedLogitModel(), FixedLogitModel(), inputs)

    def test_rmu_activation_loss_normalizes_each_example_independently(self):
        trainer = object.__new__(RMU)
        activations = torch.ones(2, 4, 3, requires_grad=True)
        targets = torch.zeros_like(activations)
        mask = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        )

        loss = trainer.compute_activation_loss(activations, targets, mask)

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))
        loss.backward()
        self.assertTrue(torch.equal(activations.grad[0, 2:], torch.zeros(2, 3)))
        self.assertGreater(activations.grad[0, :2].abs().sum().item(), 0.0)

    def test_rmu_activation_loss_rejects_empty_examples(self):
        trainer = object.__new__(RMU)
        activations = torch.zeros(2, 3, 4)
        mask = torch.tensor(
            [
                [True, False, False],
                [False, False, False],
            ]
        )

        with self.assertRaisesRegex(ValueError, "at least one non-masked token"):
            trainer.compute_activation_loss(activations, activations, mask)

    def test_rmu_forward_hook_is_removed_when_model_forward_fails(self):
        class FailingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 2)

            def forward(self, input_ids):
                self.layer(input_ids)
                raise RuntimeError("forward failed")

        trainer = object.__new__(RMU)
        model = FailingModel()

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            trainer.forward_with_cache(
                model,
                {"input_ids": torch.ones(1, 2)},
                model.layer,
                no_grad=False,
            )

        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_undial_adjusts_only_non_masked_answer_tokens(self):
        class FixedLogitModel(nn.Module):
            def __init__(self, logits, trainable=False):
                super().__init__()
                if trainable:
                    self.logits = nn.Parameter(logits.clone())
                else:
                    self.register_buffer("logits", logits.clone())
                self.input_keys = set()

            def forward(self, **inputs):
                self.input_keys = set(inputs)
                return SimpleNamespace(logits=self.logits)

        student_logits = torch.tensor(
            [[[1.0, 0.0, -1.0], [8.0, -8.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]]
        )
        teacher_logits = torch.tensor(
            [[[0.5, 1.0, -0.5], [2.0, 3.0, 4.0], [1.5, -0.5, 0.5], [0.0, 0.0, 0.0]]]
        )
        student = FixedLogitModel(student_logits, trainable=True)
        teacher = FixedLogitModel(teacher_logits)
        labels = torch.tensor([[-100, 1, -100, 2]])
        inputs = {
            "input_ids": torch.tensor([[4, 5, 6, 7]]),
            "labels": labels,
            "pixel_values": torch.randn(1, 3, 2, 2),
        }

        loss, outputs = compute_undial_loss(student, teacher, inputs, beta=2.0)

        adjusted_teacher = teacher_logits[0, [0, 2]].clone()
        adjusted_teacher[0, 1] -= 2.0
        adjusted_teacher[1, 2] -= 2.0
        expected = -(
            adjusted_teacher.softmax(dim=-1)
            * student_logits[0, [0, 2]].log_softmax(dim=-1)
        ).sum(dim=-1).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertIs(outputs.logits, student.logits)
        self.assertIn("pixel_values", student.input_keys)
        self.assertIn("pixel_values", teacher.input_keys)

        loss.backward()
        self.assertTrue(torch.equal(student.logits.grad[0, 1], torch.zeros(3)))
        self.assertTrue(torch.equal(student.logits.grad[0, 3], torch.zeros(3)))

    def test_mmunlearner_saliency_mask_uses_forget_to_preserve_ratio(self):
        forget = {"layer": torch.tensor([2.0, 1.0, 0.5])}
        preserve = {"layer": torch.tensor([1.0, 2.0, 0.5])}
        mask = compute_saliency_mask(forget, preserve, threshold=1.0)
        self.assertTrue(torch.equal(mask["layer"], torch.tensor([True, False, True])))

    def test_manu_combines_metrics_and_prunes_output_rows(self):
        forget = {
            "I_abs": {"layer": torch.tensor([4.0, 1.0, 2.0])},
            "I_var": {"layer": torch.tensor([2.0, 1.0, 2.0])},
            "I_rms": {"layer": torch.tensor([2.0, 1.0, 2.0])},
        }
        retain = {
            "I_abs": {"layer": torch.tensor([1.0, 1.0, 1.0])},
            "I_var": {"layer": torch.tensor([1.0, 1.0, 1.0])},
            "I_rms": {"layer": torch.tensor([1.0, 1.0, 1.0])},
        }
        importance = compute_combined_importance(forget, retain)
        masks = compute_topk_masks(importance, prune_percent=34)
        self.assertTrue(torch.equal(masks["layer"], torch.tensor([True, False, False])))

        linear = nn.Linear(2, 3, bias=True)
        linear.weight.data.fill_(1.0)
        linear.bias.data.fill_(1.0)
        pruned = apply_output_neuron_masks({"layer": linear}, masks)
        self.assertEqual(pruned, 1)
        self.assertTrue(torch.equal(linear.weight[0], torch.zeros(2)))
        self.assertEqual(linear.bias[0].item(), 0.0)
        self.assertTrue(torch.equal(linear.weight[1], torch.ones(2)))

    def test_manu_qwen_vision_forward_uses_image_grid(self):
        class FakeQwenVision(nn.Module):
            def forward(self, hidden_states, grid_thw):
                self.hidden_states = hidden_states
                self.grid_thw = grid_thw
                return hidden_states

        vision = FakeQwenVision()
        pixel_values = torch.randn(4, 3, 2, 2)
        image_grid_thw = torch.tensor([[1, 2, 2]])
        output = MANU._vision_forward(
            vision,
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            architecture="qwen2_5_vl",
        )
        self.assertIs(output, pixel_values)
        self.assertIs(vision.hidden_states, pixel_values)
        self.assertIs(vision.grid_thw, image_grid_thw)


if __name__ == "__main__":
    unittest.main()
