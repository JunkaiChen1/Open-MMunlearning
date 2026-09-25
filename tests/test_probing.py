import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "probing_stress_test",
        REPO_ROOT / "scripts" / "probing_stress_test.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProbingRunnerTest(unittest.TestCase):
    def test_dry_run_builds_train_and_eval_commands_for_each_layer(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            forget_data = temporary_path / "forget.parquet"
            forget_data.touch()
            output_dir = temporary_path / "result"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "probing_stress_test.py"),
                "--benchmark",
                "mllmu",
                "--model",
                "probed-Qwen2.5-VL-3B-Instruct",
                "--checkpoint",
                "/models/unlearned-checkpoint",
                "--layers",
                "8",
                "16",
                "--forget-data",
                str(forget_data),
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
            manifest = json.loads(
                (output_dir / "probing_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["layers"], [8, 16])
            self.assertEqual(manifest["input_checkpoint_kind"], "full_model")
            self.assertIn("train", manifest["commands"]["8"])
            self.assertIn("eval", manifest["commands"]["8"])
            self.assertIn("~eval", manifest["commands"]["8"]["train"])
            self.assertIn("model.model_args.n_layers=16", completed.stdout)
            self.assertIn(
                "+model.model_args.reinitialize_output_head=false",
                completed.stdout,
            )

    def test_adapter_is_merged_before_head_only_training(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapter_dir = Path(temporary_directory) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/models/base"}),
                encoding="utf-8",
            )

            kind, overrides = script._checkpoint_overrides(str(adapter_dir))

            self.assertEqual(kind, "adapter")
            self.assertIn(
                "model.model_args.pretrained_model_name_or_path=/models/base",
                overrides,
            )
            self.assertIn("+model.merge_adapter=true", overrides)

    def test_collects_flattened_multimodal_summary_metric(self):
        script = _load_script()
        scores = script._collect_scores(
            {"MLLMU_SUMMARY.json": {"forget/generation/ROUGE": 0.75}},
            ["MLLMU_SUMMARY.json:forget/generation/ROUGE"],
        )
        self.assertEqual(scores[0]["value"], 0.75)


if __name__ == "__main__":
    unittest.main()
