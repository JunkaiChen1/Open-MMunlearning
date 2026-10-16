"""SafeEraser refusal rate and its SD-image variant."""

from collections import Counter

from evals.metrics import register_metric

from .base import unlearning_metric


def refusal_statistics(counts, *, sd=False):
    total = sum(counts.values())
    if sd:
        return {
            "sd_refuse_rate": counts["refuse"] / total if total else None,
            "sd_refuse_count": counts["refuse"],
            "total_sd_rr_queries": total,
            "invalid_sarr_count": counts["None"],
        }
    return {
        "refuse_rate": counts["refuse"] / total if total else None,
        "refuse_count": counts["refuse"],
        "total_rr_queries": total,
        "invalid_rr_count": counts["None"],
    }


@register_metric
@unlearning_metric("refusal_rate")
def refusal_rate(model, *, pre_compute, access_key="judgements", **kwargs):
    counts = Counter(pre_compute[access_key]["value_by_index"])
    summary = refusal_statistics(counts)
    return {"agg_value": summary["refuse_rate"], "value_by_index": dict(counts)}
