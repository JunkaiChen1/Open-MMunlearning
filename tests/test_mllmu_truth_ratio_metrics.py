import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_evaluator():
    for name in list(sys.modules):
        if name == "evals" or name.startswith("evals."):
            del sys.modules[name]
    package = types.ModuleType("evals")
    package.__path__ = [str(SRC / "evals")]
    sys.modules["evals"] = package
    base = types.ModuleType("evals.base")
    base.Evaluator = object
    sys.modules["evals.base"] = base
    previous_transformers = sys.modules.get("transformers")
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = type("AutoProcessor", (), {})
    sys.modules["transformers"] = transformers
    path = SRC / "evals" / "mllmu.py"
    spec = importlib.util.spec_from_file_location("evals.mllmu", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evals.mllmu"] = module
    spec.loader.exec_module(module)
    if previous_transformers is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = previous_transformers
    return module


class MLLMUTruthRatioMetricsTest(unittest.TestCase):
    def test_generation_truth_ratio_and_distribution_metrics(self):
        module = _load_evaluator()
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "variants.jsonl"
            rows = []
            for row_id, modality in (("p1", "Image_Textual"), ("p1", "Pure_Text")):
                rows.append(
                    {
                        "id": row_id,
                        "question_index": len(rows),
                        "modality": modality,
                        "paraphrased_answer": f"positive {modality}",
                        "perturbed_answers": [
                            f"wrong one {modality}",
                            f"wrong two {modality}",
                            f"wrong three {modality}",
                        ],
                    }
                )
            sidecar.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            evaluator = module.MLLMUBenchEvaluator(
                OmegaConf.create(
                    {
                        "generation_truth_ratio": True,
                        "generation_truth_ratio_path": str(sidecar),
                        "generation_answer_probability": False,
                        "generation_mia": False,
                        "generation_mia_min_k": False,
                        "save_task_details": True,
                    }
                )
            )
            evaluator._row_image = lambda row, mode: None
            evaluator._build_prompt = (
                lambda text, image, shots, answer=None, **kwargs: (
                    text if answer is None else f"{text} ANSWER={answer}",
                    [],
                )
            )
            evaluator._generate = lambda *args, **kwargs: "generated"

            def answer_loss(*args, **kwargs):
                full_prompt = args[3]
                if "ANSWER=positive" in full_prompt:
                    loss = 1.0
                elif "ANSWER=wrong" in full_prompt:
                    loss = 3.0
                else:
                    loss = 1.0
                return {"loss": loss, "avg_loss": loss, "num_tokens": 1}

            evaluator._answer_loss = answer_loss
            frame = pd.DataFrame(
                [
                    {
                        "ID": "p1",
                        "Generation_Task": [
                            {
                                "Type": "Image_Textual",
                                "Question": "Image question",
                                "Ground_Truth": "ground truth image",
                            },
                            {
                                "Type": "Pure_Text",
                                "Question": "Text question",
                                "Ground_Truth": "ground truth text",
                            },
                        ],
                    }
                ]
            )
            result = evaluator._evaluate_generation(frame, object(), object(), "forget")
            self.assertEqual(result["counts"]["image_textual_truth_ratio_total"], 1)
            self.assertEqual(result["counts"]["pure_text_truth_ratio_total"], 1)
            self.assertAlmostEqual(
                result["Average Truth Ratio (Image_Textual)"], math.exp(-2)
            )
            self.assertAlmostEqual(
                result["Average Truth Ratio (Pure_Text)"], math.exp(-2)
            )

            logs = {
                "forget": {"generation": result},
                "retain_shared": {
                    "generation": {
                        "truth_ratio_distribution": {
                            "Image_Textual": [1.0, 1.1],
                            "Pure_Text": [1.0, 1.1],
                        }
                    }
                },
            }
            evaluator._add_generation_distribution_metrics(
                logs, ["forget", "retain_shared"]
            )
            generation = logs["forget"]["generation"]
            self.assertIn("KS Test PValue (Image_Textual)", generation)
            self.assertIn("KS Test Statistic (Pure_Text)", generation)
            self.assertIn("JS Distance (Image_Textual)", generation)
            self.assertEqual(
                generation["distribution_reference_split"], "retain_shared"
            )


if __name__ == "__main__":
    unittest.main()
