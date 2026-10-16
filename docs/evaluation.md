# Creating Evaluations

Benchmark evaluators in [src/evals/](../src/evals/) handle data loading, prompts, inference, and protocol-specific execution. Files in [src/evals/metrics/](../src/evals/metrics/) own calculations and attack implementations. YAML configurations choose metrics and their dependencies.

<!-- benchmark evaluator 负责数据、提示词、推理和协议流程，metrics/ 负责指标计算及攻击实现，YAML 选择指标与依赖。 -->

- [Add a Metric](#add-a-metric)
- [Input Requirements](#input-requirements)
- [Reference Logs and Recalculation](#reference-logs-and-recalculation)
- [Add a Benchmark](#add-a-benchmark)
- [Run an Example](#run-an-example)

<!-- 本指南介绍新指标、输入需求、参考日志、重新计算、新 benchmark 和运行示例。 -->

## Add a Metric

### Implement and Register

Create one file per calculation. For example, `src/evals/metrics/prediction_length.py` can measure the mean generated answer length in characters:

<!-- 每种计算方式单独放入一个指标文件，下例新增 prediction_length.py，计算生成答案的平均字符数。 -->

```python
from evals.metrics import register_metric
from evals.metrics.base import unlearning_metric
from evals.metrics.utils import generation_samples


@register_metric
@unlearning_metric("prediction_length")
def prediction_length(model, *, pre_compute, **kwargs):
    samples = generation_samples(pre_compute["generation"])
    values = {
        index: {"score": float(len(str(sample["prediction"])))}
        for index, sample in samples.items()
    }
    scores = [item["score"] for item in values.values()]
    return {
        "agg_value": sum(scores) / len(scores) if scores else None,
        "value_by_index": values,
    }
```

<!-- 两个装饰器包装并注册指标函数，agg_value 是整体结果，value_by_index 是逐样本结果，空输入返回 None。字符数仅用于演示新增计算方式。 -->

Import the new module in [src/evals/metrics/__init__.py](../src/evals/metrics/__init__.py) so its decorators execute during package loading:

<!-- 在指标包初始化文件中导入新模块，使包加载时执行注册装饰器。 -->

```python
from . import prediction_length  # noqa: F401
```

<!-- 不需要另设 scorer 目录，也不需要在数据集 evaluator 中复制指标公式。 -->

### Declare the Inference Dependency

For MLLMU generation, create `configs/eval/mllmu_metrics/generation_prediction_length.yaml`:

<!-- 将新指标接入 MLLMU 生成任务时，创建以下指标 YAML。 -->

```yaml
# @package eval.mllmu.metrics.generation_prediction_length

defaults:
  - .@pre_compute.generation_pass: generation_pass

handler: prediction_length
task: generation
pre_compute:
  generation_pass:
    access_key: generation
```

<!-- generation_pass 负责推理并提供预测记录，access_key 将依赖结果映射到 pre_compute["generation"]；包声明决定指标在合成配置中的位置。 -->

`pre_compute` declares other registered metrics to execute first. `generation_samples` accepts existing saved-generation layouts and returns records with unique IDs, predictions, and references. Another benchmark can reuse the calculation if its inference dependency supplies this contract; collecting those inputs remains the evaluator's responsibility.

<!-- pre_compute 声明先执行的依赖指标；generation_samples 将已有日志转为包含唯一 ID、预测与参考答案的记录。其他基准提供兼容输入后即可复用计算，采集输入仍由 evaluator 负责。 -->

### Include the Metric in an Experiment

Create `configs/experiment/examples/mllmu_prediction_length.yaml`:

<!-- 创建实验 YAML，将新指标加入已有评估流程。 -->

```yaml
# @package _global_

defaults:
  - /experiment/eval/mllmu/default
  - /eval/mllmu_metrics: generation_prediction_length
  - _self_

eval:
  mllmu:
    tasks: [generation]
    generation_truth_ratio: false
```

<!-- 示例继承 MLLMU 原模型评估并添加指标，仅运行生成任务，关闭需要额外文件的 Truth Ratio。 -->

```bash
python src/eval.py experiment=examples/mllmu_prediction_length
```

<!-- 完成上述文件和注册后，通过统一评估入口运行。 -->

For a working example already in the repository, see [exact_match.py](../src/evals/metrics/exact_match.py), [generation_exact_match.yaml](../configs/eval/mllmu_metrics/generation_exact_match.yaml), and [mllmu_extra_metric.yaml](../configs/experiment/examples/mllmu_extra_metric.yaml). New names in this guide are templates to create, rather than bundled components.

<!-- 仓库已有可运行的 Exact Match 实现、指标配置与实验配置；本指南中的新名称是需要自行创建的模板，不是内置组件。 -->

## Input Requirements

MLLMU generation collects additional inputs according to its configured metric dependencies. Declare required fields on the decorator or through the metric YAML's `required_inputs`. Other benchmarks need their own collection support for these fields.

<!-- MLLMU 生成流程根据指标依赖采集额外输入，可通过装饰器或 YAML 的 required_inputs 声明需求；其他基准也必须实现对应采集逻辑。 -->

| Input | Used by |
| --- | --- |
| `avg_loss` | LOSS and answer-probability metrics |
| `token_log_probs` | Min-K |
| `token_log_probs`, `mu`, `sigma` | Min-K++ |
| `grad_norm` | GradNorm; configure `p` on the metric |
| `target_loss`, `ref_loss` | Reference Loss; configure a compatible reference model |
| `base_loss`, `perturbation_losses` | Truth Ratio; needs matching answer variants |

<!-- 输入分别对应答案平均 loss、答案 token 对数概率、Min-K++ 分布统计、梯度范数、目标与参考模型 loss，以及真实或扰动答案 loss。 -->

Use the existing [input-metric experiment](../configs/experiment/examples/mllmu_input_metrics.yaml) and [reference-model experiment](../configs/experiment/examples/mllmu_reference_metric.yaml) as templates. A shared MLLMU generation pass uses one GradNorm `p` and one reference-model configuration.

<!-- 输入指标与参考模型实验已有模板，一次共享生成流程使用同一组 GradNorm p 和参考模型配置。 -->

For loss and token statistics, reuse the benchmark's answer masking and image-token alignment in [inference.py](../src/evals/utils/inference.py). Existing metrics provide examples of normalization and aggregation; keep their calculation conventions when reusing them.

<!-- loss 和 token 统计可复用现有答案掩码与图像 token 对齐实现；归一化和聚合可参考已有指标，复用时保留其计算约定。 -->

## Reference Logs and Recalculation

`reference_logs` loads explicit JSON inputs before calling a metric. Each named entry has a `path` and can optionally select keys through `include`. A null path is skipped; the handler defines how to handle an absent reference. See [forget_quality.yaml](../configs/eval/fiubench_metrics/forget_quality.yaml).

<!-- reference_logs 在指标执行前加载指定 JSON，可通过 include 选择其中的键；空路径跳过加载，缺失参考时的行为由 handler 决定。可参考 FIUBench 的 forget_quality 配置。 -->

Evaluation recomputes predictions and scores on each run. Dependencies share results only within the current evaluation scope. Use separate cache scopes for different splits or tasks, and distinct dependency names when parameters differ.

<!-- 每次评估重新计算预测与指标，仅在当前作用域内共享依赖结果；不同划分或任务需使用不同 cache 作用域，参数不同的依赖应使用不同名称。 -->

## Add a Benchmark

### Implement the Evaluator

Create `src/evals/mybench.py`, subclass [Evaluator](../src/evals/base.py), and prepare samples for the configured metrics. This skeleton separates protocol-specific inference from metric execution:

<!-- 新建 mybench.py 并继承 Evaluator，为指标准备输入。以下骨架将协议专用推理与指标执行分开。 -->

```python
from evals.base import Evaluator


class MyBenchEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("MyBench", eval_cfg, **kwargs)

    def collect_samples(self, model, split):
        raise NotImplementedError("Implement the benchmark's data and inference protocol")

    def _evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        model = self.prepare_model(model)
        output_dir = output_dir or self.eval_cfg.output_dir
        logs = {}
        for split in self.eval_cfg.splits:
            samples = self.collect_samples(model, split)
            split_logs = {}
            for name, result in self.evaluate_metrics(
                model,
                scope=split,
                task="generation",
                split=split,
                samples=samples,
            ):
                self.merge_metric_result(split_logs, name, result)
            logs[split] = split_logs
        summary = self.summarize(logs)
        self.save_result(output_dir, logs, summary)
        return summary
```

<!-- collect_samples 需按官方协议实现数据读取、提示模板、生成与参考答案提取；每条记录含唯一 id、prediction 和 reference。evaluate_metrics 执行指标，基类保存完整日志与摘要。 -->

Implement `collect_samples` before running the template. Use [mllmu.py](../src/evals/mllmu.py), [fiubench.py](../src/evals/fiubench.py), or [safeeraser.py](../src/evals/safeeraser.py) as protocol examples. If all metric handlers load their own inputs, the base evaluator can execute the suite without a custom `_evaluate`.

<!-- 该模板必须先实现 collect_samples，协议流程可参考已有 evaluator；如果所有指标 handler 都自行加载输入，也可直接使用基类流程而不重写 _evaluate。 -->

### Expose Shared Predictions

Create `src/evals/metrics/mybench_generation.py` and import it from the metrics package so calculations can consume the collected records:

<!-- 新增并导入轻量依赖 handler，将 evaluator 采集的记录提供给后续指标。 -->

```python
from evals.metrics import register_metric
from evals.metrics.base import metric_result, unlearning_metric


@register_metric
@unlearning_metric("mybench_generation_pass")
def mybench_generation_pass(model, *, samples, **kwargs):
    return metric_result({"samples": samples})
```

<!-- 此 handler 只传递共享推理结果，评分公式仍由各指标文件实现。 -->

### Register and Configure

Add `("evals.mybench", "MyBenchEvaluator")` to the registration list in [src/evals/__init__.py](../src/evals/__init__.py). Then create `configs/eval/mybench.yaml`:

<!-- 将 evaluator 加入 evals 包初始化文件的注册列表，再新增基准评估 YAML。 -->

```yaml
# @package eval.mybench

handler: MyBenchEvaluator
output_dir: ${paths.output_dir}
splits: [forget, retain]
metrics:
  prediction_length:
    handler: prediction_length
    task: generation
    pre_compute:
      generation_pass:
        handler: mybench_generation_pass
        access_key: generation
```

<!-- 复用前文的 prediction_length，依赖新基准自己的预测记录，不把 MLLMU 专用推理流程直接用于新数据集。 -->

Create `configs/experiment/eval/mybench/default.yaml` to select the model and suite:

<!-- 新增实验 YAML，组合模型与评估套件。 -->

```yaml
# @package _global_

defaults:
  - override /model: llava-1.5-7b-hf
  - override /eval: mybench
  - _self_

task_name: mybench_${model_name}
```

<!-- 基础配置提供模型名与输出根目录，实验配置选择新 benchmark；正式实验还需补齐 collect_samples 使用的数据路径和生成参数。 -->

```bash
python src/eval.py experiment=eval/mybench/default
```

<!-- 完成 evaluator、两个指标 handler 和配置后，通过该命令运行新 benchmark。 -->

## Run an Example

You can preview a configuration or run the existing Exact Match example:

<!-- 可以先预览配置，也可直接运行已有的 Exact Match 示例。 -->

```bash
python src/eval.py experiment=examples/mllmu_extra_metric --cfg job
python src/eval.py experiment=examples/mllmu_extra_metric \
  eval.mllmu.generation_truth_ratio=false
```

<!-- 第一条命令仅展示已有示例的配置，第二条执行评估并关闭需要额外文件的 Truth Ratio；新组件接入完成后替换为对应实验名称。 -->
