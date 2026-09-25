import unittest

import torch

from attacks.jailbreak import JailbreakAttack, JailbreakConfig


class TestJailbreak(unittest.TestCase):
    def test_templates_are_answer_free_and_named(self):
        attack = JailbreakAttack(
            processor=object(),
            config=JailbreakConfig(
                variants=("prefix_injection", "affirmative_suffix", "role_playing")
            ),
        )
        for variant in attack.config.variants:
            prompt = attack.build_prompt("Which city is listed?", variant)
            self.assertIn("Which city is listed?", prompt)
            self.assertNotIn("San Francisco", prompt)

    def test_text_and_image_prompt_formats(self):
        attack = JailbreakAttack(processor=object())
        text_prompt = attack._prompt("Question", False)
        image_prompt = attack._prompt("Question", True)
        self.assertNotIn("<image>", text_prompt)
        self.assertIn("<image>", image_prompt)

    def test_attack_evaluation_runs_each_variant_once(self):
        class Processor:
            def __call__(self, **kwargs):
                return {"input_ids": torch.tensor([[1, 2]])}

            def batch_decode(self, generated, skip_special_tokens=True):
                return ["target"]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.zeros(1))
                self.calls = 0

            def generate(self, **kwargs):
                self.calls += 1
                return torch.tensor([[1, 2, 3]])

        model = Model()
        result = JailbreakAttack(Processor()).evaluate(
            model,
            [
                {
                    "id": "1",
                    "task_type": "Pure_Text",
                    "question": "q",
                    "answer": "target",
                }
            ],
        )
        self.assertEqual(model.calls, 3)
        self.assertEqual(result["num_samples"], 1)
        self.assertEqual(result["jailbreak_success_at_b"], 100.0)


if __name__ == "__main__":
    unittest.main()
