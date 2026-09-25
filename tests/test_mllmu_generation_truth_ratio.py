import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_mllmu_generation_truth_ratio.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "build_mllmu_generation_truth_ratio", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_records_extracts_generation_tasks(tmp_path):
    source = tmp_path / "full.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "ID": "person-1",
                "Generation_Task": [
                    {
                        "Question": "What is the profession?",
                        "Ground_Truth": "The person is a scientist.",
                        "Type": "Image_Textual",
                    },
                    {
                        "Question": "Where do they live?",
                        "Ground_Truth": "The person lives in Wellington.",
                        "Type": "Pure_Text",
                    },
                ],
            }
        ]
    )
    pq.write_table(table, source)

    records = MODULE.source_records(source)

    assert len(records) == 2
    assert records[0]["id"] == "person-1"
    assert records[0]["question_index"] == 0
    assert records[0]["modality"] == "Image_Textual"
    assert records[1]["question_index"] == 1
    assert MODULE.source_records(source, limit=1) == records[:1]


def test_validate_response_normalizes_valid_payload():
    result = MODULE.validate_response(
        {
            "attribute": " profession ",
            "paraphrased_answer": "The individual's profession is scientist.",
            "perturbed_answers": [
                "The person is a teacher. ",
                "The person is an engineer.",
                "The person is a lawyer.",
            ],
        },
        "The person is a scientist.",
        3,
    )

    assert result["attribute"] == "profession"
    assert result["perturbed_answers"][0] == "The person is a teacher."


@pytest.mark.parametrize(
    ("paraphrase", "answers"),
    [
        (
            "The individual's profession is scientist.",
            [
                "The person is a scientist.",
                "The person is a teacher.",
                "The person is a lawyer.",
            ],
        ),
        (
            "The individual's profession is scientist.",
            [
                "The person is a teacher.",
                " the person is a teacher. ",
                "The person is a lawyer.",
            ],
        ),
        (
            "The individual's profession is scientist.",
            ["The person is a teacher."],
        ),
        (
            " the person is a scientist. ",
            [
                "The person is a teacher.",
                "The person is an engineer.",
                "The person is a lawyer.",
            ],
        ),
        (
            "The individual's profession is scientist.",
            [
                {"answer": "The person is a teacher."},
                "The person is an engineer.",
                "The person is a lawyer.",
            ],
        ),
    ],
)
def test_validate_response_rejects_invalid_perturbations(paraphrase, answers):
    with pytest.raises(ValueError):
        MODULE.validate_response(
            {
                "attribute": "profession",
                "paraphrased_answer": paraphrase,
                "perturbed_answers": answers,
            },
            "The person is a scientist.",
            3,
        )


def test_cache_key_tracks_prompt_model_and_source():
    record = {
        "id": "person-1",
        "question_index": 0,
        "modality": "Pure_Text",
        "question": "What is the profession?",
        "ground_truth": "The person is a scientist.",
    }

    first = MODULE.cache_key(record, "model-a", 3)
    assert first == MODULE.cache_key(record, "model-a", 3)
    assert first != MODULE.cache_key(record, "model-b", 3)
    assert first != MODULE.cache_key(record, "model-a", 4)
