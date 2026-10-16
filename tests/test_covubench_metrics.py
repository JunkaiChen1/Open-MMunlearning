import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evals.covubench import (  # noqa: E402
    CoVUBenchEvaluator,
    keyword_recall,
    parse_keywords,
    semantic_dissimilarity_scores,
)


class CoVUBenchMetricTest(unittest.TestCase):
    def test_parse_keywords_flattens_json_values(self):
        keywords = parse_keywords('{"name":"Kipp", "traits":["curious", "glib"]}')
        self.assertEqual(keywords, ["Kipp", "curious", "glib"])
        self.assertEqual(parse_keywords("{}"), [])

    def test_keyword_recall_is_case_insensitive_phrase_recall(self):
        prediction = "KIPP uses Symbiotic Cartography, but nothing else."
        keywords = {
            "name": "Kipp",
            "abilities": ["Symbiotic Cartography", "Echo Step"],
        }
        self.assertAlmostEqual(keyword_recall(prediction, keywords), 2 / 3)
        self.assertIsNone(keyword_recall(prediction, {}))

    def test_semantic_dissimilarity_uses_clamped_zero_to_hundred_scale(self):
        predictions = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        answers = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(
            semantic_dissimilarity_scores(predictions, answers),
            [0.0, 100.0, 100.0],
        )

    def test_summary_reports_paper_scale_and_raw_fractions(self):
        evaluator = CoVUBenchEvaluator(OmegaConf.create({"rouge_use_stemmer": True}))
        logs = {
            "splits": {
                "forget": {
                    "completed": True,
                    "divergence_completed": True,
                    "samples": {
                        "f": {
                            "keyword_recall": 0.25,
                            "semantic_dissimilarity": 80.0,
                            "question_type": "single_modal",
                            "type": "character",
                        }
                    },
                },
                "test": {
                    "completed": True,
                    "samples": {
                        "t": {
                            "keyword_recall": 0.5,
                            "question_type": "single_modal",
                            "type": "character",
                        }
                    },
                },
                "retain": {
                    "completed": True,
                    "samples": {
                        "r": {
                            "keyword_recall": 0.75,
                            "rougeL_recall": 0.6,
                            "question_type": "single_modal",
                            "type": "character",
                        }
                    },
                },
            }
        }
        summary = evaluator.summarize(logs)
        self.assertEqual(summary["Efficacy"], 75.0)
        self.assertEqual(summary["Generality"], 50.0)
        self.assertEqual(summary["Divergence"], 80.0)
        self.assertEqual(summary["Fluency"], 60.0)
        self.assertEqual(summary["Specificity"], 75.0)
        self.assertEqual(summary["raw_metrics"]["forget_keyword_em"], 0.25)

    def test_split_evaluation_resumes_without_regenerating_saved_samples(self):
        evaluator = CoVUBenchEvaluator(
            OmegaConf.create({"rouge_use_stemmer": True, "checkpoint_interval": 1})
        )
        rows = [
            {"_sample_id": "forget:000000"},
            {"_sample_id": "forget:000001"},
            {"_sample_id": "forget:000002"},
        ]
        generated_ids = []
        checkpoints = []
        evaluator._selected_count = lambda split, concepts: len(rows)
        evaluator._iter_split_rows = lambda split, concepts: iter(rows)

        def sample_log(row, model, tokenizer, split):
            generated_ids.append(row["_sample_id"])
            return {"sample_id": row["_sample_id"]}

        evaluator._sample_log = sample_log
        split_log = {
            "samples": {
                "forget:000000": {"sample_id": "forget:000000"},
            }
        }
        evaluator._evaluate_split(
            "forget",
            split_log,
            set(),
            model=None,
            tokenizer=None,
            checkpoint=lambda: checkpoints.append(len(split_log["samples"])),
        )
        self.assertEqual(generated_ids, ["forget:000001", "forget:000002"])
        self.assertEqual(split_log["processed_count"], 3)
        self.assertTrue(split_log["completed"])
        self.assertEqual(checkpoints, [2, 3, 3])


if __name__ == "__main__":
    unittest.main()
