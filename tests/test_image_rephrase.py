import unittest

import numpy as np
import torch
from PIL import Image

from attacks.image_rephrase import ImageRephraseAttack, ImageRephraseConfig


class TestImageRephrase(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(
            np.full((32, 48, 3), [80, 120, 160], dtype=np.uint8), mode="RGB"
        )
        self.record = {"id": "person:1", "image": self.image}
        self.attack = ImageRephraseAttack(
            processor=object(),
            config=ImageRephraseConfig(
                variants=("noise", "occlusion", "remove_irrelevant"),
                remove_fill="black",
            ),
        )

    def test_variants_keep_size_and_are_deterministic(self):
        for variant in self.attack.config.variants:
            first = self.attack.transform(self.image, self.record, variant)
            second = self.attack.transform(self.image, self.record, variant)
            self.assertEqual(first.size, self.image.size)
            self.assertEqual(np.asarray(first).tobytes(), np.asarray(second).tobytes())

    def test_extended_variants_keep_size(self):
        config = ImageRephraseConfig(
            variants=("brightness", "contrast", "resize", "crop", "blur"),
            brightness_factor=0.8,
            contrast_factor=0.8,
            resize_scale=0.8,
            crop_ratio=0.8,
            rephrase_blur_radius=1.5,
        )
        attack = ImageRephraseAttack(processor=object(), config=config)
        source = Image.fromarray(
            np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3), mode="RGB"
        )
        for variant in config.variants:
            transformed = attack.transform(source, {"id": "extended"}, variant)
            self.assertEqual(transformed.size, source.size)
            self.assertFalse(
                np.array_equal(np.asarray(transformed), np.asarray(source))
            )

    def test_remove_mask_is_applied(self):
        mask = np.zeros((32, 48), dtype=np.uint8)
        mask[:, :8] = 1
        record = dict(self.record, remove_mask=mask)
        transformed = self.attack.transform(self.image, record, "remove_irrelevant")
        self.assertFalse(
            np.array_equal(
                np.asarray(transformed)[:, :8], np.asarray(self.image)[:, :8]
            )
        )

    def test_regenerated_image_requires_explicit_source(self):
        with self.assertRaises(ValueError):
            self.attack.transform(self.image, self.record, "regenerate")
        generated = Image.new("RGB", self.image.size, "white")
        transformed = self.attack.transform(
            self.image, dict(self.record, regenerated_image=generated), "regenerate"
        )
        self.assertEqual(transformed.getpixel((0, 0)), (255, 255, 255))

    def test_evaluate_does_not_run_clean_baseline(self):
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

        attack = ImageRephraseAttack(
            Processor(),
            config=ImageRephraseConfig(
                variants=("noise", "occlusion"), save_images=False
            ),
        )
        model = Model()
        result = attack.evaluate(
            model,
            [{"id": "x", "image": self.image, "question": "q", "answer": "target"}],
        )
        self.assertEqual(model.calls, 2)
        self.assertNotIn("clean_target_recovery_rate", result)
        self.assertNotIn("attack_gain", result["variants"]["noise"])


if __name__ == "__main__":
    unittest.main()
