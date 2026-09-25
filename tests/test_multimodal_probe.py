import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import load_merged_peft_adapter  # noqa: E402
from model.probe import _configure_probe_model  # noqa: E402


class _Decoder(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(n_layers))
        self.config = type("Config", (), {"num_hidden_layers": n_layers})()


class _MultimodalProbeModel(nn.Module):
    def __init__(self, architecture, n_layers=4):
        super().__init__()
        self.architecture = architecture
        self.vision_tower = nn.Linear(2, 2)
        self.multi_modal_projector = nn.Linear(2, 2)
        self.visual = self.vision_tower
        self.language_model = nn.Module()
        self.language_model.model = _Decoder(n_layers)
        self.model = _Decoder(n_layers)
        self.lm_head = nn.Linear(2, 5, bias=False)
        self.config = type(
            "Config",
            (),
            {
                "num_hidden_layers": n_layers,
                "tie_word_embeddings": True,
                "text_config": type(
                    "TextConfig",
                    (), {"num_hidden_layers": n_layers, "tie_word_embeddings": True},
                )(),
            },
        )()

    def get_decoder(self):
        if self.architecture in {"llava", "llava_next", "gemma3"}:
            return self.language_model.model
        return self.model

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    @staticmethod
    def _init_weights(module):
        module.weight.data.fill_(0.25)


class MultimodalProbeTest(unittest.TestCase):
    def test_probe_preserves_multimodal_modules_for_each_supported_architecture(self):
        for architecture in ("llava", "llava_next", "qwen2_5_vl", "gemma3"):
            with self.subTest(architecture=architecture):
                model = _MultimodalProbeModel(architecture)
                vision_weight = model.vision_tower.weight.detach().clone()
                projector_weight = model.multi_modal_projector.weight.detach().clone()

                _configure_probe_model(
                    model,
                    n_layers=2,
                    freeze_base_model=True,
                    head_pretrained_model_name_or_path=None,
                    reinitialize_output_head=True,
                    reference_model_class=object,
                    load_kwargs={},
                )

                self.assertEqual(len(model.get_decoder().layers), 2)
                self.assertEqual(model.get_decoder().config.num_hidden_layers, 2)
                self.assertEqual(model.config.text_config.num_hidden_layers, 2)
                self.assertFalse(model.config.tie_word_embeddings)
                self.assertFalse(model.config.text_config.tie_word_embeddings)
                self.assertTrue(torch.equal(model.vision_tower.weight, vision_weight))
                self.assertTrue(
                    torch.equal(model.multi_modal_projector.weight, projector_weight)
                )
                self.assertFalse(model.vision_tower.weight.requires_grad)
                self.assertFalse(model.multi_modal_projector.weight.requires_grad)
                self.assertTrue(model.lm_head.weight.requires_grad)
                self.assertTrue(torch.all(model.lm_head.weight == 0.25))

    def test_probe_rejects_non_positive_layer_counts(self):
        for n_layers in (0, -1, True, 1.5):
            with self.subTest(n_layers=n_layers):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _configure_probe_model(
                        _MultimodalProbeModel("llava"),
                        n_layers=n_layers,
                        freeze_base_model=True,
                        head_pretrained_model_name_or_path=None,
                        reinitialize_output_head=True,
                        reference_model_class=object,
                        load_kwargs={},
                    )

    def test_probe_can_preserve_a_trained_output_head_for_evaluation(self):
        model = _MultimodalProbeModel("llava")
        original_head = model.get_output_embeddings().weight.detach().clone()

        _configure_probe_model(
            model,
            n_layers=2,
            freeze_base_model=True,
            head_pretrained_model_name_or_path=None,
            reinitialize_output_head=False,
            reference_model_class=object,
            load_kwargs={},
        )

        self.assertTrue(
            torch.equal(original_head, model.get_output_embeddings().weight)
        )

    def test_merged_adapter_keeps_only_the_probe_head_trainable(self):
        model = _MultimodalProbeModel("llava")
        merged_adapter = type(
            "MergedAdapter",
            (),
            {"merge_and_unload": lambda self: model},
        )()

        with patch(
            "peft.PeftModel.from_pretrained", return_value=merged_adapter
        ) as load_adapter:
            merged_model = load_merged_peft_adapter(model, "/models/adapter")

        load_adapter.assert_called_once_with(
            model, "/models/adapter", is_trainable=False
        )
        self.assertIs(merged_model, model)
        self.assertTrue(model.lm_head.weight.requires_grad)
        self.assertFalse(model.vision_tower.weight.requires_grad)
        self.assertFalse(model.multi_modal_projector.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
