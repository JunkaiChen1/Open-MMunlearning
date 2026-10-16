import json
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evals.multimodal_benchmarks import MultimodalBenchmarkEvaluator  # noqa: E402


def _config(directory, benchmark, name, filename, fmt):
    return OmegaConf.create(
        {
            "name": name,
            "benchmark": benchmark,
            "format": fmt,
            "data_path": str(Path(directory) / filename),
            "image_root": str(directory),
            "generation_args": {},
            "overwrite": True,
        }
    )


def test_pope_loader_and_metrics():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pope.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "question": "Is there a cat?",
                        "answer": answer,
                        "category": "random",
                        "image": "missing.jpg",
                    }
                )
                for answer in ("yes", "no", "yes", "no")
            )
            + "\n",
            encoding="utf-8",
        )
        evaluator = MultimodalBenchmarkEvaluator(
            _config(directory, "pope", "POPE", "pope.jsonl", "jsonl")
        )
        evaluator._generate = lambda model, tokenizer, record: {
            "yes": "yes",
            "no": "no",
        }[record["answer"]]
        result = evaluator.evaluate(object(), output_dir=directory, overwrite=True)
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0


def test_mmbench_options_and_category_accuracy():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mmbench.tsv"
        path.write_text(
            "index\tquestion\tA\tB\tanswer\tl2-category\n"
            "1\tWhich?\tone\ttwo\tB\tperception\n",
            encoding="utf-8",
        )
        evaluator = MultimodalBenchmarkEvaluator(
            _config(directory, "mmbench", "MMBench", "mmbench.tsv", "tsv")
        )
        evaluator._generate = lambda model, tokenizer, record: "The answer is B."
        result = evaluator.evaluate(object(), output_dir=directory, overwrite=True)
        assert result["accuracy"] == 1.0
        assert result["category_accuracy"] == {"perception": 1.0}


def test_mmvet_json_mapping_and_text_scores():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mm-vet.json"
        path.write_text(
            json.dumps(
                {
                    "sample-1": {
                        "question": "What is shown?",
                        "answer": "a red car",
                        "capability": "rec",
                        "image": "missing.jpg",
                    }
                }
            ),
            encoding="utf-8",
        )
        evaluator = MultimodalBenchmarkEvaluator(
            _config(directory, "mmvet", "MM-Vet", "mm-vet.json", "json")
        )
        class RougeStub:
            def score(self, prediction, reference):
                return type("Score", (), {"value": 1.0})()

        evaluator._rouge = RougeStub()
        evaluator._generate = lambda model, tokenizer, record: "A red car."
        result = evaluator.evaluate(object(), output_dir=directory, overwrite=True)
        assert result["accuracy"] == 1.0
        assert result["rougeL"] > 0.9
        assert result["category_accuracy"] == {"rec": 1.0}


def test_mmvet_alternative_and_conjunctive_answers():
    evaluator = MultimodalBenchmarkEvaluator(
        OmegaConf.create({"name": "MM-Vet", "benchmark": "mmvet"})
    )
    assert evaluator._mmvet_match("5/4", "1.25<OR>=1.25<OR>5/4") == 1.0
    assert evaluator._mmvet_match("red car", "red<AND>car") == 1.0
    assert evaluator._mmvet_match("red", "red<AND>car") == 0.0


def test_vizwiz_consensus_score_and_answer_list_parsing():
    evaluator = MultimodalBenchmarkEvaluator(
        OmegaConf.create({"name": "VizWiz", "benchmark": "vizwiz"})
    )
    references = evaluator._answer_list(
        '["a cat", "cat", "cat", "dog", "chair", "table", '
        '"room", "pet", "animal", "kitten"]'
    )
    assert len(references) == 10
    assert evaluator._vizwiz_score("the cat", references) == 0.9
    assert evaluator._vizwiz_score("dog", references) == 0.3
    assert evaluator._vizwiz_score("sofa", references) == 0.0


def test_gqa_parquet_join_and_short_answer_score():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        import pandas as pd

        pd.DataFrame(
            [{"id": "q1", "imageId": "img1", "question": "Is it red?", "answer": "no"}]
        ).to_parquet(directory / "questions.parquet")
        pd.DataFrame(
            [{"id": "img1", "image": {"bytes": b"not-an-image"}}]
        ).to_parquet(directory / "images.parquet")
        evaluator = MultimodalBenchmarkEvaluator(
            OmegaConf.create(
                {
                    "name": "GQA",
                    "benchmark": "gqa",
                    "data_path": str(directory / "questions.parquet"),
                    "images_path": str(directory / "images.parquet"),
                }
            )
        )
        record = evaluator._records()[0]
        assert record["id"] == "q1"
        assert record["answer"] == "no"
        assert record["image"]["bytes"] == b"not-an-image"
        assert evaluator._score_record(record, "No") == (1.0, "no", "no")


def test_vqav2_join_and_soft_score():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "questions.json").write_text(
            json.dumps(
                {
                    "questions": [
                        {"question_id": 7, "image_id": 42, "question": "What color?"}
                    ]
                }
            ),
            encoding="utf-8",
        )
        (directory / "annotations.json").write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "question_id": 7,
                            "multiple_choice_answer": "blue",
                            "answer_type": "other",
                            "answers": [{"answer": answer} for answer in ["blue", "blue", "blue", "red"]],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        evaluator = MultimodalBenchmarkEvaluator(
            OmegaConf.create(
                {
                    "name": "VQAv2",
                    "benchmark": "vqav2",
                    "data_path": str(directory / "questions.json"),
                    "annotations_path": str(directory / "annotations.json"),
                }
            )
        )
        record = evaluator._records()[0]
        assert record["image"] == "val2014/COCO_val2014_000000000042.jpg"
        assert len(record["references"]) == 4
        assert evaluator._score_record(record, "blue")[0] == 0.75


def test_seeded_sample_selection_is_reproducible():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        path = directory / "rows.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"id": str(index), "question": f"q{index}", "answer": "yes"})
                for index in range(20)
            )
            + "\n",
            encoding="utf-8",
        )

        def records(seed):
            evaluator = MultimodalBenchmarkEvaluator(
                OmegaConf.create(
                    {
                        "name": "sampled",
                        "benchmark": "custom",
                        "format": "jsonl",
                        "data_path": str(path),
                        "max_samples": 5,
                        "sample_seed": seed,
                    }
                )
            )
            return [record["id"] for record in evaluator._records()]

        first = records(17)
        assert first == records(17)
        assert first != records(18)
        assert len(first) == 5
