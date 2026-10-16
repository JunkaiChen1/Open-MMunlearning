"""Source-level protocol contracts for the multimodal benchmark evaluators.

These tests deliberately avoid loading a model or the project's full evaluator
registry.  Optional ``transformers``/DeepSpeed initialization is replaced with
small stubs, while the actual evaluator methods and benchmark data are used.
"""

import hashlib
import importlib.util
import math
import sys
import types
import unittest
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


try:
    from omegaconf import OmegaConf

    HAVE_OMEGACONF = True
except ImportError:  # pragma: no cover - lightweight environments
    HAVE_OMEGACONF = False


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HAS_FIUBENCH_FIXTURES = (
    ROOT / "data" / "eval" / "fiubench" / "upstream_split.json"
).is_file() and (ROOT / "external_repos" / "FIUBench" / "dataset" / "split.json").is_file()
HAS_CLEAR_FIXTURES = (
    ROOT / "data" / "eval" / "clear" / "forget10_perturbed" / "train-00000-of-00001.parquet"
).is_file()


def _load_evaluator(stem):
    """Load one evaluator without importing evals.__init__ or model classes."""
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

    path = SRC / "evals" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"evals.{stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"evals.{stem}"] = module
    spec.loader.exec_module(module)
    if previous_transformers is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = previous_transformers
    return module


@unittest.skipUnless(HAVE_OMEGACONF, "omegaconf is not installed")
class MultimodalUpstreamContractTest(unittest.TestCase):
    def test_mllmu_raw_prompt_and_lower_answer_contract(self):
        module = _load_evaluator("mllmu")
        evaluator = module.MLLMUBenchEvaluator(
            OmegaConf.create(
                {
                    "prompt_style": "raw",
                    "model_family": "llava",
                    "upstream_compatibility": True,
                    "few_shot": {},
                }
            )
        )
        evaluator._prompt_style = "raw"
        evaluator._active_model_family = "llava"
        shot = {
            "text": "Question: What?\nA: one\nB: two",
            "answer": "B",
            "image": None,
        }
        prompt, images = evaluator._build_prompt(
            "Question: Now?\nA: yes\nB: no", None, [shot]
        )
        self.assertEqual(
            prompt,
            "USER:\nQuestion: What?\nA: one\nB: two\nCorrect Answer: B\n"
            "USER:\nQuestion: Now?\nA: yes\nB: no\nASSISTANT:",
        )
        self.assertEqual(images, [])
        generation_prompt, _ = evaluator._build_prompt(
            "Describe it", object(), [], raw_assistant_prefix="ASSISTANT: "
        )
        self.assertTrue(generation_prompt.endswith("\nASSISTANT: "))
        text_generation_prompt, _ = evaluator._build_prompt(
            "Describe it", None, [], text_user_prefix="USER: "
        )
        self.assertEqual(text_generation_prompt, "USER: Describe it\nASSISTANT:")
        blank_shot = {
            "text": "Q [Blank]",
            "answer": "answer",
            "image": "shot-image",
            "raw_image_prefix": "USER:<image>\n",
        }
        blank_prompt, blank_images = evaluator._build_prompt(
            "Current [Blank]", "current-image", [blank_shot]
        )
        self.assertEqual(
            blank_prompt,
            "USER:<image>\nQ [Blank]\nCorrect Answer: answer\n"
            "USER: <image>\nCurrent [Blank]\nASSISTANT:",
        )
        self.assertEqual(blank_images, ["shot-image", "current-image"])
        self.assertEqual(evaluator._clean_answer("USER: x\nASSISTANT: B"), "B")
        self.assertEqual(
            evaluator._contains_scorer.score("A  B", "a  b").value, 1.0
        )

    def test_mllmu_can_disable_few_shot_without_loading_demonstrations(self):
        module = _load_evaluator("mllmu")
        evaluator = module.MLLMUBenchEvaluator(
            OmegaConf.create(
                {
                    "use_few_shot": False,
                    "few_shot": {
                        "classification": {"default": 1},
                        "fill_in_the_blank": {"default": 2},
                    },
                }
            )
        )

        def fail_if_loaded():
            self.fail("Few-shot data must not load when use_few_shot=false.")

        evaluator._load_few_shot_df = fail_if_loaded
        self.assertFalse(evaluator._use_few_shot())
        self.assertEqual(evaluator._few_shot_count("classification", "llava"), 0)
        self.assertEqual(evaluator._few_shot_count("fill_in_the_blank", "llava"), 0)
        self.assertEqual(
            evaluator._few_shots_for_classification(["sample"], "llava"),
            ([], [], {}),
        )
        self.assertEqual(
            evaluator._few_shots_for_blank(["sample"], "llava"),
            ([], [], {}),
        )

    def test_mllmu_source_input_adapters_and_generation_calls(self):
        module = _load_evaluator("mllmu")
        evaluator = module.MLLMUBenchEvaluator(
            OmegaConf.create(
                {
                    "prompt_style": "raw",
                    "model_family": "llava",
                    "upstream_compatibility": True,
                    "few_shot": {},
                }
            )
        )

        class Processor:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return {"input_ids": torch.tensor([[1, 2]])}

        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[3, 4]])}

            def decode(self, ids, **kwargs):
                return "A"

        processor = Processor()
        tokenizer = Tokenizer()
        evaluator.processor = processor
        evaluator._prepare_inputs("x", ["image"], tokenizer, input_mode="processor")
        self.assertEqual(processor.calls[-1]["images"], ["image"])
        evaluator._prepare_inputs(
            "x", ["image"], tokenizer, input_mode="processor", single_image=True
        )
        self.assertEqual(processor.calls[-1]["images"], "image")
        evaluator._prepare_inputs("x", [], tokenizer, input_mode="processor")
        self.assertIsNone(processor.calls[-1]["images"])
        evaluator._prepare_inputs("x", ["image"], tokenizer, input_mode="tokenizer")
        self.assertEqual(len(processor.calls), 3)

        calls = []
        evaluator._few_shots_for_classification = lambda ids, family: ([], [], {})
        evaluator._row_image = lambda row, mode: None
        evaluator._build_prompt = lambda *args, **kwargs: ("prompt", [])
        evaluator._generate = lambda *args, **kwargs: calls.append(kwargs) or "A"
        frame = pd.DataFrame(
            [
                {
                    "ID": "person",
                    "Classification_Task": {
                        "Image_Textual_Questions": [
                            {
                                "Question": "Q1",
                                "Options": {"A": "yes"},
                                "Correct_Answer": "A",
                            }
                        ],
                        "Pure_Text_Questions": [
                            {
                                "Question": "Q2",
                                "Options": {"A": "yes"},
                                "Correct_Answer": "A",
                            }
                        ],
                    },
                }
            ]
        )
        evaluator._evaluate_classification(
            frame, ["person"], object(), tokenizer, "llava", "forget"
        )
        self.assertEqual(calls[0]["input_mode"], "processor")
        self.assertNotIn("max_new_tokens", calls[0])
        self.assertEqual(calls[1]["input_mode"], "tokenizer")
        self.assertIsNone(calls[1]["max_new_tokens"])

        calls.clear()
        evaluator._active_model_family = "llava"
        evaluator._prompt_style = "raw"
        evaluator._row_image = lambda row, mode: "current-image"
        evaluator._generate = lambda *args, **kwargs: calls.append(kwargs) or "answer"
        generation_frame = pd.DataFrame(
            [
                {
                    "ID": "person",
                    "Generation_Task": [
                        {
                            "Type": "Image_Textual",
                            "Question": "Image question",
                            "Ground_Truth": "answer",
                        },
                        {
                            "Type": "Pure_Text",
                            "Question": "Text question",
                            "Ground_Truth": "answer",
                        },
                    ],
                }
            ]
        )
        evaluator._evaluate_generation(generation_frame, object(), tokenizer, "forget")
        self.assertTrue(calls[0]["single_image"])
        self.assertEqual(calls[0]["input_mode"], "processor")
        self.assertFalse(calls[1]["single_image"])
        self.assertEqual(calls[1]["input_mode"], "tokenizer")

    def test_mllmu_generation_mia_covers_image_and_text_questions(self):
        module = _load_evaluator("mllmu")
        evaluator = module.MLLMUBenchEvaluator(
            OmegaConf.create(
                {
                    "prompt_style": "raw",
                    "model_family": "llava",
                    "upstream_compatibility": True,
                    "generation_mia": True,
                    "generation_mia_sign": 1.0,
                    "generation_mia_min_k": True,
                    "save_task_details": True,
                }
            )
        )
        evaluator._prompt_style = "raw"
        evaluator._active_model_family = "llava"
        evaluator._row_image = lambda row, mode: "image"
        evaluator._generate = lambda *args, **kwargs: "generated answer"
        evaluator._bleu = lambda ground_truth, generated: 0.25

        class Rouge:
            def score(self, prediction, reference):
                return types.SimpleNamespace(
                    details={"rouge1": 0.1, "rouge2": 0.2, "rougeL": 0.3}
                )

        evaluator._generation_rouge_scorer = Rouge()
        loss_calls = []

        def answer_loss(
            model,
            tokenizer,
            prompt,
            full_prompt,
            images,
            *,
            input_mode,
            single_image,
            compute_token_stats=False,
        ):
            loss_calls.append((prompt, full_prompt, input_mode, single_image))
            average_loss = 1.5 if input_mode == "processor" else 2.5
            token_log_probs = (
                torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0])
                if input_mode == "processor"
                else torch.tensor([-2.0, -1.0])
            )
            return {
                "loss": average_loss * 2,
                "avg_loss": average_loss,
                "num_tokens": 2,
                "token_log_probs": token_log_probs,
                "mu": torch.zeros_like(token_log_probs),
                "sigma": torch.ones_like(token_log_probs),
            }

        evaluator._answer_loss = answer_loss
        frame = pd.DataFrame(
            [
                {
                    "ID": "person",
                    "Generation_Task": [
                        {
                            "Type": "Image_Textual",
                            "Question": "Image question",
                            "Ground_Truth": "image truth",
                        },
                        {
                            "Type": "Pure_Text",
                            "Question": "Text question",
                            "Ground_Truth": "text truth",
                        },
                    ],
                }
            ]
        )

        result = evaluator._evaluate_generation(frame, object(), object(), "forget")

        self.assertEqual([call[2] for call in loss_calls], ["processor", "tokenizer"])
        self.assertEqual([call[3] for call in loss_calls], [True, False])
        self.assertTrue(loss_calls[0][1].endswith("image truth"))
        self.assertTrue(loss_calls[1][1].endswith("text truth"))
        self.assertAlmostEqual(result["Average Loss MIA (Image_Textual)"], 1.5)
        self.assertAlmostEqual(result["Average Loss MIA (Pure_Text)"], 2.5)
        self.assertAlmostEqual(
            result["Average Answer Probability (Image_Textual)"], math.exp(-1.5)
        )
        self.assertAlmostEqual(
            result["Average Answer Probability (Pure_Text)"], math.exp(-2.5)
        )
        self.assertAlmostEqual(
            result["Average ZLIB MIA (Image_Textual)"],
            1.5 / len(zlib.compress(b"image truth")),
        )
        self.assertAlmostEqual(
            result["Average ZLIB MIA (Pure_Text)"],
            2.5 / len(zlib.compress(b"text truth")),
        )
        self.assertEqual(result["Average Min-K 20% MIA (Image_Textual)"], 5.0)
        self.assertEqual(
            result["Average Min-K++ 20% MIA (Image_Textual)"], 5.0
        )
        self.assertEqual(result["Average Min-K 20% MIA (Pure_Text)"], 2.0)
        self.assertEqual(result["Average Min-K++ 20% MIA (Pure_Text)"], 2.0)
        self.assertEqual(result["counts"]["image_textual_mia_total"], 1)
        self.assertEqual(result["counts"]["pure_text_mia_total"], 1)
        self.assertEqual(result["counts"]["image_textual_probability_total"], 1)
        self.assertEqual(result["counts"]["pure_text_probability_total"], 1)
        self.assertEqual(result["counts"]["image_textual_min_k_mia_total"], 1)
        self.assertEqual(result["counts"]["pure_text_min_k_mia_total"], 1)
        self.assertIn("loss_mia", result["details"][0])
        self.assertIn("zlib_mia", result["details"][1])
        self.assertAlmostEqual(result["details"][0]["answer_probability"], math.exp(-1.5))
        self.assertAlmostEqual(result["details"][1]["answer_probability"], math.exp(-2.5))
        self.assertIn("min_k_20_mia", result["details"][0])
        self.assertIn("min_k_plus_plus_20_mia", result["details"][1])

    def test_mllmu_teacher_forced_loss_masks_prompt_tokens(self):
        module = _load_evaluator("mllmu")
        evaluator = module.MLLMUBenchEvaluator(OmegaConf.create({}))

        class Tokenizer:
            pad_token_id = 0
            padding_side = "right"

            def __call__(self, text, return_tensors="pt"):
                length = len(str(text))
                return {
                    "input_ids": torch.arange(1, length + 1).unsqueeze(0),
                    "attention_mask": torch.ones((1, length), dtype=torch.long),
                }

        class Processor:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.image_calls = []

            def __call__(self, text, images=None, return_tensors="pt"):
                self.image_calls.append(images)
                return self.tokenizer(text, return_tensors=return_tensors)

            def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=True
            ):
                self.template_call = (messages, add_generation_prompt)
                return "templated answer"

        class Model:
            def __init__(self):
                self.parameter = torch.nn.Parameter(torch.zeros(()))
                self.labels = []

            def parameters(self):
                yield self.parameter

            def __call__(self, **kwargs):
                self.labels.append(kwargs["labels"].detach().cpu())
                return types.SimpleNamespace(loss=torch.tensor(2.0))

        tokenizer = Tokenizer()
        processor = Processor(tokenizer)
        model = Model()
        evaluator.processor = processor

        pure_text = evaluator._answer_loss(
            model,
            tokenizer,
            "abc",
            "abcdef",
            [],
            input_mode="tokenizer",
        )
        image_text = evaluator._answer_loss(
            model,
            tokenizer,
            "abc",
            "abcdef",
            ["image"],
            input_mode="processor",
            single_image=True,
        )

        self.assertEqual(pure_text["num_tokens"], 3)
        self.assertEqual(pure_text["avg_loss"], 2.0)
        self.assertEqual(pure_text["loss"], 6.0)
        self.assertEqual(image_text, pure_text)
        self.assertTrue(torch.equal(model.labels[0][0, :3], torch.full((3,), -100)))
        self.assertEqual(model.labels[0][0, 3:].tolist(), [4, 5, 6])
        self.assertEqual(processor.image_calls, ["image", "image"])

        evaluator._prompt_style = "chat_template"
        evaluator._build_prompt("question", answer="ground truth")
        messages, add_generation_prompt = processor.template_call
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"][0]["text"], "ground truth")
        self.assertFalse(add_generation_prompt)

    @unittest.skipUnless(
        HAS_FIUBENCH_FIXTURES,
        "FIUBench evaluation fixtures are external to the code-only checkout",
    )
    def test_fiubench_uses_upstream_split_and_llava_phi_labels(self):
        module = _load_evaluator("fiubench")
        data_root = ROOT / "data" / "eval" / "fiubench"
        evaluator = module.FIUBenchEvaluator(
            OmegaConf.create(
                {
                    "data_root": str(data_root),
                    "split_path": str(data_root / "upstream_split.json"),
                    "max_people": 400,
                    "upstream_compatibility": True,
                    "model_family": "llava-phi",
                    "prompt_style": "raw",
                }
            )
        )
        full_path = str(data_root / "full.json")
        self.assertEqual(
            hashlib.sha256(
                (data_root / "upstream_split.json").read_bytes()
            ).hexdigest(),
            hashlib.sha256(
                (
                    ROOT / "external_repos" / "FIUBench" / "dataset" / "split.json"
                ).read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(len(evaluator._filtered_records(full_path, "forget5")), 20)
        self.assertEqual(len(evaluator._filtered_records(full_path, "retain5")), 20)
        evaluator._active_model_family = "llava-phi"
        evaluator._active_prompt_style = "raw"
        self.assertTrue(evaluator._uses_upstream_llava_io())
        evaluator._active_model_family = "llava"
        self.assertFalse(evaluator._uses_upstream_llava_io())
        evaluator._active_model_family = "llava-phi"
        self.assertEqual(
            evaluator._raw_prompt_fields(),
            {
                "system_prompt": "",
                "question_start": "<|user|>\n",
                "answer_tag": "<|end|>\n<|assistant|>\n",
            },
        )

        class Tokenizer:
            pad_token_id = 0
            padding_side = "right"

            def __call__(self, text):
                return {"input_ids": [10, 11, 12, 13, 14, 15]}

        tokenizer = Tokenizer()
        full_batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]])}
        labels = evaluator._make_labels(
            full_batch,
            full_batch,
            tokenizer,
            full_text="<|user|>\nQ<|end|>\n<|assistant|>\nanswer",
        )
        self.assertTrue(torch.equal(labels[0, :5], torch.full((5,), -100)))
        self.assertEqual(labels[0, 5:].tolist(), [6, 7])

        probabilities = torch.arange(1, 11, dtype=torch.float64) / 10
        loss_result = {
            "token_log_probs": probabilities.log(),
            "mu": torch.zeros(10, dtype=torch.float64),
            "sigma": torch.ones(10, dtype=torch.float64),
        }
        min_k, min_k_plus_plus = evaluator._min_k_scores(loss_result)
        expected_min_k = sum(
            probabilities[:k].log().mean().exp().item() * weight
            for k, weight in zip(range(1, 6), [0.3, 0.3, 0.2, 0.1, 0.1])
        )
        self.assertAlmostEqual(min_k, expected_min_k)
        self.assertAlmostEqual(min_k_plus_plus, expected_min_k)

        average_loss = 2.5
        answer = "answer with repeated words"
        compressed_length = len(zlib.compress(answer.encode("utf-8")))
        self.assertAlmostEqual(
            evaluator._loss_mia_scorer.score(average_loss).value,
            -average_loss,
        )
        self.assertAlmostEqual(
            evaluator._zlib_mia_scorer.score(average_loss, answer).value,
            -average_loss / compressed_length,
        )

        task = {
            "avg_gt_loss": {"0": 1.0, "1": 2.0},
            "average_perturb_loss": {"0": [2.0, 3.0], "1": [3.0, 4.0]},
            "avg_paraphrased_loss": {"0": 1.0, "1": 2.0},
            "rougeL_recall": {"0": 0.2, "1": 0.4},
        }
        utility = evaluator._upstream_model_utility(
            {"eval_forget_log.json": task, "eval_retain_log.json": task}
        )
        expected_forget_probability = (math.exp(-1) + math.exp(-2)) / 2
        expected_retain_probability = (
            math.exp(-1) / (math.exp(-1) + math.exp(-2) + math.exp(-3))
            + math.exp(-2) / (math.exp(-2) + math.exp(-3) + math.exp(-4))
        ) / 2
        self.assertAlmostEqual(utility["Prob. Forget"], expected_forget_probability)
        self.assertAlmostEqual(utility["Prob. Retain"], expected_retain_probability)

    @unittest.skipUnless(
        HAS_CLEAR_FIXTURES,
        "CLEAR evaluation fixtures are external to the code-only checkout",
    )
    def test_clear_source_questions_caps_logits_and_rows(self):
        module = _load_evaluator("clear")
        data_root = ROOT / "data" / "eval" / "clear"
        evaluator = module.CLEAREvaluator(
            OmegaConf.create(
                {
                    "data_root": str(data_root),
                    "image_root": str(data_root),
                    "upstream_compatibility": True,
                    "max_samples_per_task": 300,
                    "auto_perturbations": False,
                    "prompt_style": "chat_template",
                    "seed": 0,
                }
            )
        )

        class ExpandedTokenizer:
            pad_token_id = 0
            padding_side = "right"

            def tokenize(self, text, add_special_tokens=True):
                # The textual image marker is one token, while the processor
                # batch below has already expanded it to four positions.
                return ["question", "answer-prefix"]

        expanded_tokenizer = ExpandedTokenizer()
        expanded_full = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "attention_mask": torch.ones((1, 8), dtype=torch.long),
        }
        expanded_prompt = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
        }
        expanded_labels = evaluator._make_labels(
            expanded_full,
            expanded_prompt,
            expanded_tokenizer,
            prompt_text="<image> question",
        )
        self.assertTrue(
            torch.equal(expanded_labels[0, :6], torch.full((6,), -100))
        )
        self.assertEqual(expanded_labels[0, 6:].tolist(), [7, 8])

        self.assertEqual(
            tuple(module.CAPTION_QUESTIONS),
            (
                "What can you see in this picture?",
                "Tell me about the content of this image",
                "Can you give a description of the image?",
                "What is depicted in the image?",
                "Explain what you observe in the picture.",
                "Describe the image in detail.",
                "What is the main subject of this image?",
                "Can you describe the scene or objects in the image?",
                "What is happening in this image?",
            ),
        )
        spec = {
            "eval_task": "eval_log_forget",
            "split": "forget10_perturbed",
            "data_path": str(
                data_root / "forget10_perturbed" / "train-00000-of-00001.parquet"
            ),
            "question_strategy": "random_caption",
            "caption_questions": module.CAPTION_QUESTIONS,
            "answer_key": "caption",
            "base_answer_key": "paraphrased_caption",
            "perturbed_answer_key": "perturbed_captions",
        }
        samples = evaluator._samples_for_task(spec)
        self.assertEqual(len(samples), 300)
        self.assertIsNone(samples[0]["question"])
        self.assertEqual(samples[0]["question_strategy"], "random_caption")

        class DeterministicChoice:
            def __init__(self):
                self.calls = 0

            def choice(self, values):
                value = values[self.calls % len(values)]
                self.calls += 1
                return value

        evaluator.rng = DeterministicChoice()
        self.assertNotEqual(
            evaluator._question_for_sample(samples[0]),
            evaluator._question_for_sample(samples[0]),
        )

        class Model:
            class Config:
                image_token_id = 99

            config = Config()

        input_ids = torch.tensor([[1, 99, 2, 3]])
        logits = torch.arange(14, dtype=torch.float32).reshape(1, 7, 2)
        aligned = evaluator._remove_image_tokens(input_ids, logits, Model())
        expected = torch.cat([logits[0, :1], logits[0, -3:]], dim=0).unsqueeze(0)
        self.assertTrue(torch.equal(aligned, expected))

        class IndexModel:
            class Config:
                image_token_index = 99

            config = Config()

        aligned_by_index = evaluator._remove_image_tokens(
            input_ids, logits, IndexModel()
        )
        self.assertTrue(torch.equal(aligned_by_index, expected))

    def test_clear_model_utility_matches_source_formula(self):
        module = _load_evaluator("clear")
        evaluator = module.CLEAREvaluator(
            OmegaConf.create({"include_extended_group_metrics": False})
        )
        task = {
            "avg_gt_loss": {"0": 1.0, "1": 2.0},
            "average_perturb_loss": {"0": [2.0, 3.0], "1": [3.0, 4.0]},
            "avg_paraphrased_loss": {"0": 1.0, "1": 2.0},
            "rougeL_recall": {"0": 0.2, "1": 0.4},
        }
        result = evaluator._compute_model_utility(
            {"eval_log.json": task, "eval_log_forget.json": task}
        )
        self.assertIn("Model Utility", result)
        self.assertNotIn("Real metric", result)
        self.assertAlmostEqual(result["ROUGE Retain"], 0.3)
        expected_probability = (math.exp(-1) + math.exp(-2)) / 2
        self.assertAlmostEqual(result["Prob. Retain"], expected_probability)
        self.assertAlmostEqual(
            evaluator._probability_metric("eval_real_faces_wo_options.json", task),
            (
                math.exp(-1) / (math.exp(-1) + math.exp(-2) + math.exp(-3))
                + math.exp(-2) / (math.exp(-2) + math.exp(-3) + math.exp(-4))
            )
            / 2,
        )
        # Each sample averages its own perturbation losses before applying the
        # forget-mode truth-ratio transform.
        expected_forget_truth_ratio = math.exp(-1.5)
        self.assertAlmostEqual(
            result["Truth Ratio Forget"], expected_forget_truth_ratio
        )

        text_task = {
            "avg_gt_loss": {"0": 1.0, "1": 2.0},
            "rougeL_recall": {"0": 0.2, "1": 0.4},
        }
        with_text = evaluator._compute_model_utility(
            {
                "eval_log.json": task,
                "eval_log_forget.json": task,
                "eval_text_retain.json": text_task,
            }
        )
        self.assertAlmostEqual(with_text["Model Utility"], result["Model Utility"])
        text_probability = (math.exp(-1) + math.exp(-2)) / 2
        expected_text_utility = 2 / (1 / text_probability + 1 / 0.3)
        self.assertAlmostEqual(with_text["Pure Text Utility"], expected_text_utility)

        if importlib.util.find_spec("scipy"):
            forget_quality = evaluator._evaluate_forget_quality(
                {"eval_log_forget.json": task},
                {"eval_log_forget.json": task},
            )
            self.assertEqual(forget_quality["KS test p-value"], 1.0)
            self.assertAlmostEqual(forget_quality["JS metric"], 0.0)

    def test_clear_truth_distribution_averages_perturbations_per_sample(self):
        module = _load_evaluator("clear")
        evaluator = module.CLEAREvaluator(OmegaConf.create({}))
        distribution = evaluator._task_truth_distribution(
            {
                "avg_paraphrased_loss": {"first": 1.0, "second": 2.0},
                "average_perturb_loss": {
                    "first": [2.0, 3.0],
                    "second": [100.0, 100.0],
                },
            }
        )
        expected = np.asarray([math.exp(1.5), math.exp(98.0)])
        np.testing.assert_allclose(distribution, expected)

    def test_clear_filters_text_rows_before_source_cap(self):
        module = _load_evaluator("clear")
        evaluator = module.CLEAREvaluator(
            OmegaConf.create(
                {
                    "upstream_compatibility": True,
                    "max_samples_per_task": 2,
                    "auto_perturbations": False,
                    "seed": 0,
                }
            )
        )
        evaluator._load_records = lambda *args: [
            {"image": "first.jpg", "caption": "image row"},
            {"image": "second.jpg", "caption": "image row"},
            {"image": None, "question": "text question 1", "answer": "answer 1"},
            {"image": None, "question": "text question 2", "answer": "answer 2"},
            {"image": None, "question": "text question 3", "answer": "answer 3"},
        ]
        spec = {
            "eval_task": "eval_text_forget",
            "data_path": "mixed+tofu",
            "split": "forget10+tofu",
            "image_mode": "text",
            "question_strategy": "column",
            "question_key": "question",
            "answer_key": "answer",
            "max_samples": 2,
        }

        samples = evaluator._samples_for_task(spec)

        self.assertEqual(
            [sample["question"] for sample in samples],
            ["text question 1", "text question 2"],
        )
        self.assertTrue(all(sample["image_value"] is None for sample in samples))

    def test_clear_derives_pure_text_splits_from_forget_ratio(self):
        module = _load_evaluator("clear")
        evaluator = module.CLEAREvaluator(
            OmegaConf.create(
                {
                    "forget_ratio": "5",
                    "pure_text_data_root": "/datasets/CLEAR",
                    "task_specs": [
                        {
                            "eval_task": "eval_text_forget",
                            "pure_text_partition": "forget",
                            "data_path": None,
                        },
                        {
                            "eval_task": "eval_text_retain",
                            "pure_text_partition": "retain",
                            "data_path": None,
                        },
                    ],
                }
            )
        )

        forget, retain = evaluator._task_specs()

        self.assertEqual(forget["split"], "forget05+tofu")
        self.assertEqual(forget["data_path"], "/datasets/CLEAR/forget05+tofu")
        self.assertEqual(retain["split"], "retain95+tofu")
        self.assertEqual(retain["data_path"], "/datasets/CLEAR/retain95+tofu")

    def test_clear_generation_writes_shared_loss_and_zlib_mia(self):
        module = _load_evaluator("clear")
        evaluator = module.CLEAREvaluator(
            OmegaConf.create(
                {
                    "generation_mia": True,
                    "generation_mia_sign": 1.0,
                    "generation_mia_min_k": True,
                    "save_generated_text": False,
                }
            )
        )
        sample = {"index": 0, "category": "Forget"}
        evaluator._samples_for_task = lambda spec: [sample]
        evaluator._evaluate_perturbation_ratio = lambda *args: {}
        evaluator._answer_values = lambda *args, **kwargs: ["ground truth"]
        evaluator._question_for_sample = lambda value: "question"
        evaluator._answer_loss = lambda *args, **kwargs: {
            "loss": 5.0,
            "avg_loss": 2.5,
            "num_tokens": 2,
            "token_log_probs": torch.tensor([-4.0, -3.0, -2.0, -1.0]),
            "mu": torch.zeros(4),
            "sigma": torch.ones(4),
        }
        evaluator._generate = lambda *args, **kwargs: ("prompt", "generated")
        evaluator._rouge_recall = lambda *args: {}

        result = evaluator._evaluate_task(
            {"eval_task": "eval_log_forget", "answer_key": "answer"},
            object(),
            object(),
        )

        self.assertEqual(result["loss_mia"], {"0": 2.5})
        self.assertAlmostEqual(
            result["zlib_mia"]["0"],
            2.5 / len(zlib.compress(b"ground truth")),
        )
        self.assertEqual(result["min_k_20_mia"], {"0": 4.0})
        self.assertEqual(result["min_k_plus_plus_20_mia"], {"0": 4.0})


if __name__ == "__main__":
    unittest.main()
