# Unlearning Benchmarks and Implemented Metrics

This document summarizes the metrics implemented by the evaluators, evaluation
configs, and evaluator registry in the current repository. "Implemented" means
that the metric can be computed through the shared evaluation entry point and
written to the result files.

The metrics are grouped by evaluation goal. `Forget` measures memory removal,
target recovery, or privacy leakage on samples to be forgotten and is usually
better when lower (or when forgetting quality is higher). `Retain` measures
preserved task ability and is usually better when higher. `Other/Aggregate`
covers test-set generalization, cross-split statistics, and combined summaries.

## Implemented Unlearning Benchmarks and Metrics

| Benchmark or suite | Forget metrics | Retain metrics | Other or aggregate metrics |
| --- | --- | --- | --- |
| **MLLMU-Bench** | Image-text and text-only classification accuracy, fill-in-the-blank accuracy, generated `ROUGE-1`, `ROUGE-2`, `ROUGE-L`, `BLEU`, and `Answer Probability`; generated-answer `Loss MIA`, `ZLIB MIA`, `Min-K 20% MIA`, and `Min-K++ 20% MIA`. | The same classification, fill-in-the-blank, and generation metrics on `retain_shared` and `retain_celebrity`. | The same metrics on the `test` split, written separately by split. |
| **CLEAR** | `avg_gt_loss`, `gt_loss`, `avg_paraphrased_loss`, `average_perturb_loss`, answer probability, `ROUGE-1 Recall`, `ROUGE-L Recall`, `Truth Ratio`, `Loss MIA`, `ZLIB MIA`, `KS Test P-Value`, and `JS Metric` on forget and text-only forget data. | Answer contrastive probability, `ROUGE-1 Recall`, `ROUGE-L Recall`, and `Truth Ratio` on retain and text-only retain data. | `Model Utility` and `Pure Text Utility` aggregate retain tasks; `Forget Quality` combines forget and reference-retain distributions. |
| **FIUBench** | `ROUGE-1 Recall`, `ROUGE-L Recall`, answer probability, `Truth Ratio`, keyword `Exact Match/APE`, `Mink`, `Mink++`, `Loss MIA`, and `ZLIB MIA`; `Prob. Forget`, `ROUGE Forget`, `Truth Ratio Forget`, `Forget Quality`, `KS Test PVal Forget`, and `KS Test Forget`. | Corresponding quality metrics plus `Prob. Retain`, `ROUGE Retain`, and `Truth Ratio Retain`; `GPT Retain` and `EM Retain` are optional. | `Model Utility` is the harmonic mean of retain metrics. |
| **CoVUBench** | `keyword_recall`, semantic dissimilarity, and aggregate `Efficacy` and `Divergence`; forget group metrics are also reported. | `keyword_recall`, `ROUGE-L Recall`, and aggregate `Fluency` and `Specificity`; retain group metrics are also reported. | `Generality` uses the `test` split; test group metrics are separate from both sides. |
| **FigStep** | `figstep_target_recovery_rate` on the configured forget split. Lower is better. | The current config does not run the retain split. | - |
| **Image Rephrase** | Each image-variant `target_recovery_rate` and `attack_success_at_b` on forget data. Lower is better. | The current config does not run the retain split. | - |
| **Jailbreak** | Each jailbreak variant's `target_recovery_rate`, task-type recovery rates, and `jailbreak_success_at_b` on forget data. Lower is better. | The current config does not run the retain split. | - |
| **SUA** | `sua_target_recovery_rate` on the configured forget split. Lower is better. | The current config does not run the retain split. | - |

MLLMU, CLEAR, FIUBench, and CoVUBench are shared unlearning evaluators.
FigStep, Image Rephrase, Jailbreak, and SUA are attack-oriented evaluation
suites. Attack metrics report target-answer recovery and normally belong to the
forget side.

## Benchmarks Not Yet Registered in the Shared Evaluator

| Benchmark | Current status |
| --- | --- |
| **UMUBench** | Forget: `AccF`, `RLF`; retain: `AccR`, `RLR`. These metrics are not yet computed by the shared evaluation flow. |
| **SafeEraser** | Forget: `Forget Quality`, `SARR`; retain: `Model Utility`. These metrics are not yet computed by the shared evaluation flow. |

## Reusable General Metrics

The following metrics are implemented in `src/evals/metrics/` and
`src/evals/scorers/`. Whether a benchmark uses one depends on its evaluator and
YAML configuration:

- Forget metrics: `exact_memorization`, `extraction_strength`, `privleak`,
  `ks_test`, and the `LOSS`, `ZLIB`, `Min-K`, and `Min-K++` MIA variants;
  `probability`, `rouge`, and `truth_ratio` on forget splits also belong here.
- Retain metrics: `probability`, `probability_w_options`, `rouge`,
  `truth_ratio`, and `classifier_prob` on retain splits, typically used to
  measure preserved ability.
- Split or aggregate metrics: `rel_diff`, `hm_aggregate`, `Reference`, and
  `GradNorm`, plus the same scorer on different splits, cannot be classified
  without the evaluator configuration.

## Evaluator Registry

The shared evaluation entry point currently registers MLLMU-Bench, FIUBench,
CLEAR, and CoVUBench. UMUBench and SafeEraser currently appear only in dataset
or training configs. The registry is defined in
[`src/evals/__init__.py`](../src/evals/__init__.py).
