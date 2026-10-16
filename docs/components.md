# Components

New components follow three steps: implement a handler, register it, and add a Hydra YAML configuration. Reuse an existing handler when only checkpoint paths, data locations, or parameters change.

<!-- 新组件按“实现 handler、注册、新增 Hydra YAML”三个步骤接入；仅更换权重、数据位置或参数时，优先复用已有 handler。 -->

- [Models](#models)
- [Unlearning Methods](#unlearning-methods)
- [Datasets and Collators](#datasets-and-collators)
- [Experiments](#experiments)
- [Metrics and Benchmarks](evaluation.md)

<!-- 本指南涵盖模型、遗忘方法、数据集、collator 和实验；指标与 benchmark 的接入见 evaluation.md。 -->

## Models

### Reuse an Existing Architecture

For another checkpoint of a supported architecture, copy its configuration under [configs/model/](../configs/model/). For example, create `configs/model/my_llava.yaml`:

<!-- 接入已有架构的另一份权重时，复制对应模型配置即可。以下示例创建 my_llava.yaml。 -->

```yaml
model_handler: LlavaForConditionalGeneration
model_args:
  pretrained_model_name_or_path: /path/to/llava-checkpoint
  torch_dtype: bfloat16
tokenizer_args:
  pretrained_model_name_or_path: ${model.model_args.pretrained_model_name_or_path}
template_args:
  apply_chat_template: true
  system_prompt: ""
  user_start_tag: "USER: "
  user_end_tag: " "
  asst_start_tag: "ASSISTANT: "
  asst_end_tag: " "
```

<!-- 复用已注册的 LLaVA handler，tokenizer 与 processor 跟随模型路径，提示模板与模型保持一致。 -->

Select it with `model=my_llava` in a training or evaluation command. Its YAML name also determines the model name used in standard output paths.

<!-- 在训练或评估命令中设置 model=my_llava 即可使用该配置，标准输出路径中的模型名也随之变化。 -->

### Register a New Architecture

A model handler uses `from_pretrained` to load the model. Put custom handlers under [src/model/](../src/model/). This wrapper illustrates the interface:

<!-- 模型 handler 通过 from_pretrained 加载模型，自定义实现放在 src/model/。下例演示接口。 -->

```python
class MyLlavaForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration.from_pretrained(*args, **kwargs)
```

<!-- 示例延迟导入模型。实际新架构应替换为自身实现，并支持 dtype、cache_dir 等加载参数。 -->

Import and register the class in [src/model/__init__.py](../src/model/__init__.py), then use its exact class name in `model_handler`:

<!-- 在模型包初始化文件中导入并注册类，再在 model_handler 中填写相同类名。 -->

```python
from model.my_model import MyLlavaForConditionalGeneration

_register_model(MyLlavaForConditionalGeneration)
```

<!-- 假设实现位于 src/model/my_model.py，注册名称取类名。 -->

For a new multimodal family, the main adaptation points are its processor, prompt template, and image inputs. See [data/collators.py](../src/data/collators.py), the selected evaluator, and [inference helpers](../src/evals/utils/inference.py).

<!-- 新多模态系列主要适配 processor、提示模板和图像输入，可参考 collator、所选 evaluator 与推理辅助代码。 -->

## Unlearning Methods

### Implement a Trainer

Create `src/trainer/unlearn/my_method.py` and extend [UnlearnTrainer](../src/trainer/unlearn/base.py). This example uses the existing gradient-ascent objective to illustrate the contract; replace it with your method's formula.

<!-- 新建方法文件并继承 UnlearnTrainer；下例仅用已有梯度上升目标演示接口，实际接入时替换为新方法公式。 -->

```python
from trainer.unlearn.base import UnlearnTrainer


class MyMethod(UnlearnTrainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = self._prepare_model_inputs(inputs["forget"])
        outputs = model(**forget_inputs)
        loss = -outputs.loss
        return (loss, outputs) if return_outputs else loss
```

<!-- 遗忘 batch 按 forget、retain 等键组织；调用模型前移除辅助字段，并兼容 Trainer 的返回值约定。 -->

### Register and Configure

Import and register the class in [src/trainer/__init__.py](../src/trainer/__init__.py):

<!-- 在 trainer 包初始化文件中导入并注册方法类。 -->

```python
from trainer.unlearn.my_method import MyMethod

_register_trainer(MyMethod)
```

<!-- 注册后的 handler 名称为 MyMethod。 -->

Create `configs/trainer/MyMethod.yaml`, inheriting common training arguments:

<!-- 新建方法 YAML，继承公共训练参数。 -->

```yaml
defaults:
  - finetune

handler: MyMethod
method_args: {}
```

<!-- args 用于 TrainingArguments，method_args 用于方法自定义构造参数；示例不需要额外参数。 -->

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/fiubench/default trainer=MyMethod
```

<!-- 完成实现与注册后，通过 trainer=MyMethod 使用已有基准实验运行。 -->

If a method uses special data, collator options, or PEFT settings, compose them in a benchmark experiment, following [manu.yaml](../configs/experiment/unlearn/fiubench/manu.yaml). Custom weight edits or hooks can be handled through the method's save and load logic.

<!-- 特殊数据、collator 或 PEFT 设置通过专用实验 YAML 组合；自定义权重修改和 hook 可在方法的保存与加载逻辑中处理。 -->

## Datasets and Collators

### Reuse a Dataset Handler

For image QA data, start with [ImageQADataset](../src/data/qa.py) and [FIUBENCH_QA_ft.yaml](../configs/data/datasets/FIUBENCH_QA_ft.yaml). Save a configuration such as the following under `configs/data/datasets/MyBench_QA_ft.yaml`:

<!-- 图像问答优先复用 ImageQADataset；以下配置假设包含 image 列，以及含 Question、Answer 字段的 metadata。 -->

```yaml
MyBench_QA_ft:
  handler: ImageQADataset
  args:
    hf_args:
      path: parquet
      data_files: ${paths.data_dir}/finetune/mybench/train-00000-of-00001.parquet
      split: train
    image_key: image
    id_key: ID
    qa_pairs_key: metadata
    qa_question_key: Question
    qa_answer_key: Answer
    max_length: 2048
    tokenize: false
```

<!-- 数据配置选择字段和路径，由训练 collator 将原始图像与问答转换为模型输入。 -->

For an incompatible format, implement a dataset under [src/data/](../src/data/), import it in [src/data/__init__.py](../src/data/__init__.py), and call `_register_data(MyDataset)`. The constructor receives configured `args` plus runtime values such as `tokenizer` and `template_args`.

<!-- 不兼容已有格式时，新增数据类并注册；构造函数接收 YAML 参数及 tokenizer、template_args 等运行时输入。 -->

### Add Data Preparation

Add conversion code under [scripts/](../scripts/) and an entry in [configs/data/download.yaml](../configs/data/download.yaml), including sources, preparation commands, and expected outputs. The setup CLI derives benchmark flags from those entries. Keep finetuning, forget/retain training, and evaluation partitions separate according to the official protocol.

<!-- 在 scripts/ 新增转换代码，在 download.yaml 声明数据源、准备命令和预期产物；setup CLI 根据配置生成基准选项。微调、遗忘与保持训练、评估数据应按官方协议区分。 -->

### Add a Collator When Needed

Collators are implemented in [src/data/collators.py](../src/data/collators.py), registered with `_register_collator` in [src/data/__init__.py](../src/data/__init__.py), and configured under [configs/collator/](../configs/collator/). Start with `DataCollatorForMultimodalQADataset` for nested forget/retain batches and answer-label masking.

<!-- collator 在 data/collators.py 实现，在 data/__init__.py 注册，在 configs/collator/ 配置；嵌套 batch 和答案掩码可参考现有多模态问答 collator。 -->

## Experiments

Experiments compose components without a separate registry. Put complete recipes under [configs/experiment/](../configs/experiment/) and inherit existing experiments where possible. See [Configuring Experiments](experiments.md) for defaults, overrides, paths, and configuration checks.

<!-- 实验直接组合组件，无需单独注册，完整流程放在 configs/experiment/，优先继承已有配置。defaults、覆盖方式、路径和配置检查见 experiments.md。 -->

## Metrics and Benchmarks

See [Creating Evaluations](evaluation.md) for metric registration, shared inference dependencies, input requirements, and new benchmark evaluators.

<!-- 指标注册、共享推理依赖、输入需求和新 benchmark evaluator 的接入见 evaluation.md。 -->
