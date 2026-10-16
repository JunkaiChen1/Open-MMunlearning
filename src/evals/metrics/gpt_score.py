"""Benchmark-specific GPT scoring and aggregation."""

from evals.metrics import register_metric

from .base import unlearning_metric


FIUBENCH_GPT_PROMPT = """You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for question-answer pairs about fictitious entities.
Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:
1. Focus on the meaningful match between the predicted answer and the correct answer.
2. Consider synonyms or paraphrases as valid matches.
3. Evaluate the correctness of the prediction compared to the answer.
4. Please do not consider the difference in sentence style between the correct answer and the predicted answer, but only judge whether the predicted answer makes sense based on factual accuracy.
5. If there is something in the predicted answer that is not in the correct answer, then it is considered to be hallucination.

The score should range from 0 to 1. A larger score means a better answer. The score should be a float number with 2 decimal places. For example, 0.51, 0.99, 0.00, 0.76, etc.
In additional to this, I would like you to be able to extract some key words from the question and the correct answer, which are considered to be the key to answering the question correctly, and a prediction tends to score higher if  the prediction is able to include these key words.
Please first output a single line containing only one value indicating the scores for the predicted answer.
In the subsequent line, please provide some key words of the question and correct answers.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

Question: {question}
Correct Answer: {answer}
Prediction: {prediction}

Outputs (include score, key words, explanation):"""


def fiubench_gpt_question(prompt, *, question_start, question_end, system, answer_tag):
    """Remove model tags in the order used by FIUBench/evaluate_util.py."""
    for tag in (question_start, question_end, system, answer_tag):
        prompt = prompt.replace(tag.replace("\n", ""), "")
    return prompt.strip(" ").strip("\n").strip(" ")


def parse_fiubench_gpt_score(response):
    """Keep FIUBench's first-line parsing and zero fallback unchanged."""
    try:
        score = response.split("\n")[0].strip(" ")
        if ":" in score:
            score = score[score.find(":") :].strip(":").strip(" ")
        if "**" in score:
            score = score.strip("**").strip(" ")
        return float(score)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def fiubench_gpt_score(judge, *, question, answer, prediction):
    """Run FIUBench's inline GPT metric, including its short-answer shortcut."""
    if len(prediction) <= 5:
        return 0.0
    response = judge(
        FIUBENCH_GPT_PROMPT.format(
            question=question, answer=answer, prediction=prediction
        )
    )
    return parse_fiubench_gpt_score(response)


def gpt_score_statistics(values):
    values = [
        float(value) if isinstance(value, (int, float)) else 0.0 for value in values
    ]
    return {
        "average_gpt_score": sum(values) / len(values) if values else None,
        "gpt_score_count": len(values),
    }


@register_metric
@unlearning_metric("gpt_score")
def gpt_score(model, *, pre_compute, access_key="judgements", **kwargs):
    values = pre_compute[access_key]["value_by_index"]
    if isinstance(values, dict):
        values = list(values.values())
    summary = gpt_score_statistics(values)
    return {"agg_value": summary["average_gpt_score"], "value_by_index": values}
