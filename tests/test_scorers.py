import importlib.util
import math
import sys
import unittest
import zlib
from pathlib import Path

import torch


# Import the scorer package directly so these unit tests do not initialize all
# benchmark evaluators and their optional model dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evals"))

from scorers import (  # noqa: E402
    AnswerProbabilityScorer,
    ContainsScorer,
    ContrastiveProbabilityScorer,
    ExactMatchScorer,
    JensenShannonDistanceScorer,
    KeywordRecallScorer,
    LossMIAScorer,
    MinKPlusPlusScorer,
    MinKScorer,
    PerturbationTruthRatioScorer,
    RougeScorer,
    SCORER_REGISTRY,
    SemanticDissimilarityScorer,
    StandardMinKPlusPlusScorer,
    StandardMinKScorer,
    TruthRatioTransformScorer,
    TwoSampleKSScorer,
    ZlibMIAScorer,
    get_scorer,
    parse_keywords,
    token_log_probability_statistics,
)


class ScorerTest(unittest.TestCase):
    def test_keyword_recall_flattens_json_and_returns_diagnostics(self):
        scorer = KeywordRecallScorer()
        result = scorer.score(
            "KIPP uses Symbiotic Cartography, but nothing else.",
            '{"name":"Kipp", "abilities":["Symbiotic Cartography", "Echo Step"]}',
        )
        self.assertEqual(parse_keywords("{}"), [])
        self.assertAlmostEqual(result.value, 2 / 3)
        self.assertEqual(result.details["matched_count"], 2)
        self.assertEqual(result.details["keyword_count"], 3)

    def test_text_match_scorers_normalize_case_and_whitespace(self):
        self.assertEqual(
            ExactMatchScorer().score("  A  B ", "a b").value,
            1.0,
        )
        self.assertEqual(
            ContainsScorer().score("The answer is Kipp.", " kipp ").value,
            1.0,
        )

    def test_legacy_lower_normalization_preserves_protocol_whitespace(self):
        # MLLMU/FIUBench call str.lower() directly; unlike the reusable
        # canonical mode, repeated whitespace remains significant.
        self.assertEqual(
            ContainsScorer(normalization="lower").score("a  b", "a b").value,
            0.0,
        )
        self.assertEqual(
            KeywordRecallScorer(normalization="lower").score("A  B", ["a  b"]).value,
            1.0,
        )

    def test_truth_ratio_is_shared_loss_formula(self):
        result = PerturbationTruthRatioScorer().score(2.0, [1.0, 3.0])
        self.assertAlmostEqual(result.value, 1.0)
        self.assertEqual(result.details["num_perturbations"], 2)

    def test_answer_and_contrastive_probabilities_use_average_losses(self):
        answer = AnswerProbabilityScorer().score(math.log(4.0))
        self.assertAlmostEqual(answer.value, 0.25)

        contrastive = ContrastiveProbabilityScorer().score(
            0.0, [math.log(2.0), math.log(2.0)]
        )
        self.assertAlmostEqual(contrastive.value, 0.5)
        self.assertEqual(contrastive.details["num_competing_answers"], 2)

        # A direct exp(-1000) calculation underflows for every answer.
        stable = ContrastiveProbabilityScorer().score(1000.0, [1000.0, 1000.0])
        self.assertAlmostEqual(stable.value, 1 / 3)

    def test_loss_mia_preserves_canonical_and_fiubench_orientations(self):
        canonical = LossMIAScorer().score(2.5)
        fiubench = LossMIAScorer(sign=-1.0).score(2.5)

        self.assertAlmostEqual(canonical.value, 2.5)
        self.assertAlmostEqual(fiubench.value, -2.5)
        self.assertEqual(canonical.details["average_loss"], 2.5)
        self.assertEqual(fiubench.details["sign"], -1.0)

    def test_zlib_mia_matches_upstream_compression_formula(self):
        text = "answer with repeated words " * 3
        compressed_length = len(zlib.compress(text.encode("utf-8")))

        canonical = ZlibMIAScorer().score(2.5, text)
        fiubench = ZlibMIAScorer(sign=-1.0).score(2.5, text)

        self.assertAlmostEqual(canonical.value, 2.5 / compressed_length)
        self.assertAlmostEqual(fiubench.value, -2.5 / compressed_length)
        self.assertEqual(canonical.details["compressed_length"], compressed_length)
        self.assertEqual(canonical.details["text_length"], len(text.encode("utf-8")))
        self.assertEqual(canonical.details["encoding"], "utf-8")

    def test_loss_mia_validates_inputs(self):
        with self.assertRaises(ValueError):
            LossMIAScorer(sign=0)
        with self.assertRaises(ValueError):
            LossMIAScorer().score(float("nan"))
        with self.assertRaises(ValueError):
            ZlibMIAScorer().score(1.0)

    def test_truth_ratio_transform_supports_protocol_orientation(self):
        scorer = TruthRatioTransformScorer()
        self.assertAlmostEqual(scorer.score(4.0, mode="forget").value, 0.25)
        self.assertAlmostEqual(scorer.score(4.0, mode="retain").value, 0.75)
        self.assertAlmostEqual(
            scorer.score(
                0.25,
                mode="retain",
                ratio_convention="base_over_perturb",
            ).value,
            0.75,
        )
        self.assertEqual(scorer.score(0.0, mode="forget").value, 0.0)

    def test_min_k_matches_fiubench_weighted_protocol(self):
        probabilities = torch.arange(1, 11, dtype=torch.float64) / 10
        token_log_probs = probabilities.log()
        component_scores = [
            probabilities[:k].log().mean().exp().item() for k in range(1, 6)
        ]
        expected = sum(
            score * weight
            for score, weight in zip(
                component_scores,
                [0.3, 0.3, 0.2, 0.1, 0.1],
            )
        )

        min_k = MinKScorer().score(token_log_probs)
        min_k_plus_plus = MinKPlusPlusScorer().score(
            token_log_probs,
            torch.zeros_like(token_log_probs),
            torch.ones_like(token_log_probs),
        )
        self.assertAlmostEqual(min_k.value, expected)
        self.assertAlmostEqual(min_k_plus_plus.value, expected)
        self.assertEqual(min_k.details["token_count"], 10)

    def test_min_k_preserves_short_answer_behavior(self):
        result = MinKScorer().score(torch.tensor([-0.5]))
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.details["valid_weight"], 0.0)
        with self.assertRaises(ValueError):
            MinKPlusPlusScorer().score([-0.5, -0.3], [0.0], [1.0, 1.0])

    def test_standard_min_k_matches_legacy_twenty_percent_attack(self):
        token_log_probs = torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0])
        min_k = StandardMinKScorer().score(token_log_probs)
        min_k_plus_plus = StandardMinKPlusPlusScorer().score(
            token_log_probs,
            torch.zeros_like(token_log_probs),
            torch.ones_like(token_log_probs),
        )

        self.assertEqual(min_k.value, 5.0)
        self.assertEqual(min_k_plus_plus.value, 5.0)
        self.assertEqual(min_k.details["ratio"], 0.2)
        self.assertEqual(min_k.details["selected_count"], 1)
        self.assertEqual(StandardMinKScorer().score([-2.0]).value, 2.0)
        self.assertEqual(StandardMinKScorer().score([]).value, 0.0)

    def test_standard_min_k_token_statistics_match_shifted_labels(self):
        logits = torch.tensor(
            [
                [2.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [-1.0, 0.0, 2.0],
                [2.0, -1.0, 0.0],
            ]
        ).unsqueeze(0)
        labels = torch.tensor([[-100, 1, 2, 0]])
        stats = token_log_probability_statistics(logits, labels)
        expected_vocab = torch.log_softmax(logits[0, :2], dim=-1)
        expected_targets = expected_vocab.gather(
            -1, torch.tensor([[1], [2]])
        ).squeeze(-1)

        self.assertTrue(torch.allclose(stats["token_log_probs"], expected_targets))
        self.assertEqual(stats["token_log_probs"].numel(), 2)

    def test_semantic_dissimilarity_accepts_torch_embeddings(self):
        scorer = SemanticDissimilarityScorer()
        values = scorer.values(
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        )
        self.assertEqual(values, [0.0, 100.0, 100.0])

    def test_registry_builds_named_scorers(self):
        expected_names = {
            "answer_probability",
            "contrastive_probability",
            "jensen_shannon_distance",
            "keyword_recall",
            "loss",
            "loss_mia",
            "min_k",
            "min_k_plus_plus",
            "standard_min_k",
            "standard_min_k_plus_plus",
            "truth_ratio_transform",
            "two_sample_ks",
            "zlib",
            "zlib_mia",
        }
        self.assertTrue(expected_names.issubset(SCORER_REGISTRY.names()))
        self.assertIsInstance(get_scorer("truth_ratio"), PerturbationTruthRatioScorer)

    @unittest.skipUnless(
        importlib.util.find_spec("scipy"),
        "scipy is not installed",
    )
    def test_distribution_scorers_return_diagnostics_and_align_inputs(self):
        ks_result = TwoSampleKSScorer().score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(ks_result.value, 1.0)
        self.assertEqual(ks_result.details["statistic"], 0.0)

        js_scorer = JensenShannonDistanceScorer(bins=4)
        direct = js_scorer.score([1.0, 2.0], [1.0, 2.0])
        histogram = js_scorer.score([0.0, 1.0], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(direct.value, 0.0)
        self.assertEqual(direct.details["alignment"], "weights")
        self.assertEqual(histogram.details["alignment"], "histogram")
        self.assertTrue(math.isfinite(histogram.value))

    @unittest.skipUnless(
        importlib.util.find_spec("rouge_score"),
        "rouge-score is not installed",
    )
    def test_rouge_returns_selected_value_and_all_variants(self):
        result = RougeScorer(
            ("rouge1", "rougeL"), aggregation="recall", primary="rougeL"
        ).score("the cat", "the cat sat")
        self.assertAlmostEqual(result.value, result.details["rougeL"])
        self.assertIn("rouge1", result.details)


if __name__ == "__main__":
    unittest.main()
