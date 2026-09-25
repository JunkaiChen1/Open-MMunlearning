import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import get_quantization_config  # noqa: E402


def _load_quantization_script():
    spec = importlib.util.spec_from_file_location(
        "quantization_stress_test",
        REPO_ROOT / "scripts" / "quantization_stress_test.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuantizationConfigTest(unittest.TestCase):
    def test_builds_4bit_nf4_config(self):
        cfg = OmegaConf.create(
            {
                "quantization": {
                    "enabled": True,
                    "bits": 4,
                    "quant_type": "nf4",
                    "compute_dtype": "bfloat16",
                    "use_double_quant": True,
                    "device_map": "auto",
                }
            }
        )
        quantization_config, load_kwargs = get_quantization_config(cfg)

        self.assertTrue(quantization_config.load_in_4bit)
        self.assertEqual(quantization_config.bnb_4bit_quant_type, "nf4")
        self.assertEqual(load_kwargs, {"device_map": "auto"})

    def test_disabled_quantization_returns_no_loader_arguments(self):
        cfg = OmegaConf.create({"quantization": {"enabled": False}})
        self.assertEqual(get_quantization_config(cfg), (None, {}))

    def test_rejects_unsupported_bit_width(self):
        cfg = OmegaConf.create({"quantization": {"enabled": True, "bits": 3}})
        with self.assertRaisesRegex(ValueError, "either 4 or 8"):
            get_quantization_config(cfg)


class QuantizationScriptTest(unittest.TestCase):
    def test_dry_run_writes_paper_protocol_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "result"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "quantization_stress_test.py"),
                "--experiment",
                "eval/fiubench/default",
                "--model",
                "Qwen2.5-VL-3B-Instruct",
                "--output-dir",
                str(output_dir),
                "--gpu",
                "0",
                "--dry-run",
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest_path = output_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertIn("quantization_stress_bnb_4bit", completed.stdout)
            self.assertIn(
                "+model.quantization.quant_type=nf4", completed.stdout
            )

    def test_robustness_accepts_flattened_multimodal_summary_keys(self):
        script = _load_quantization_script()
        result = script._robustness(
            {"FIUBENCH_SUMMARY.json": {"eval_retain_log.json/rougeL_recall/mean": 0.8}},
            {"FIUBENCH_SUMMARY.json": {"eval_retain_log.json/rougeL_recall/mean": 0.6}},
            ["FIUBENCH_SUMMARY.json:eval_retain_log.json/rougeL_recall/mean"],
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["quantization_robustness"], 0.75)


if __name__ == "__main__":
    unittest.main()
