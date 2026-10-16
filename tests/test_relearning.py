import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "scripts" / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RelearningRunnerTest(unittest.TestCase):
    def test_dry_run_uses_one_full_checkpoint_and_forget_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "result"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "relearning_stress_test.py"),
                "--benchmark",
                "fiubench",
                "--model",
                "Qwen2.5-VL-3B-Instruct",
                "--checkpoint",
                "/models/unlearned-checkpoint",
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output_dir / "relearning_manifest.json").read_text())
            self.assertEqual(manifest["input_checkpoint_kind"], "full_model")
            self.assertIn("data/datasets@data.train=FIUBENCH_QA_forget", completed.stdout)
            self.assertIn(
                "collator.DataCollatorForMultimodalQADataset.args.max_length="
                "${oc.select:model.tokenizer_args.model_max_length,2048}",
                completed.stdout,
            )
            self.assertIn(
                "model.model_args.pretrained_model_name_or_path=/models/unlearned-checkpoint",
                completed.stdout,
            )
            self.assertNotIn("retain_checkpoint", manifest)

    def test_detects_adapter_checkpoint(self):
        script = _load_script("relearning_stress_test")
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapter_dir = Path(temporary_directory) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}")
            self.assertEqual(script._checkpoint_kind(str(adapter_dir)), "adapter")
            self.assertEqual(
                script._checkpoint_overrides(str(adapter_dir), "adapter"),
                ["+model.adapter_path=" + str(adapter_dir), "peft=none"],
            )


class RelearningScoreTest(unittest.TestCase):
    def test_compares_flattened_multimodal_summary_metric(self):
        script = _load_script("relearning_score")
        result = script._metric_changes(
            {"FIUBENCH_SUMMARY.json": {"forget/rouge": 0.2}},
            {"FIUBENCH_SUMMARY.json": {"forget/rouge": 0.5}},
            ["FIUBENCH_SUMMARY.json:forget/rouge"],
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["absolute_change"], 0.3)
        self.assertAlmostEqual(result[0]["relative_change"], 1.5)


if __name__ == "__main__":
    unittest.main()
