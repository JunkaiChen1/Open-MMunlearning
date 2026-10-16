# Configuring Experiments

Run commands from the repository root. [src/train.py](../src/train.py) handles finetuning and unlearning; [src/eval.py](../src/eval.py) handles evaluation. Hydra combines the selected experiment with model, trainer, data, collator, and evaluation configurations.

<!-- 命令在仓库根目录运行，train.py 负责微调和遗忘，eval.py 负责评估；Hydra 将实验与模型、trainer、数据、collator、评估配置组合。 -->

## Example Commands

The following commands prepare FIUBench, finetune LLaVA-1.5-7B, run NPO, and evaluate the saved result:

<!-- 以下命令依次准备 FIUBench、微调 LLaVA-1.5-7B、运行 NPO，并评估遗忘结果。 -->

```bash
python setup_data.py --fiubench

python src/train.py experiment=finetune/fiubench/default \
  model=llava-1.5-7b-hf

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/fiubench/default model=llava-1.5-7b-hf \
  trainer=NPO forget_split=forget_10 retain_split=retain_90

python src/eval.py experiment=eval/fiubench/unlearned \
  model=llava-1.5-7b-hf trainer=NPO ratio=10
```

<!-- 训练和评估使用相同模型、方法与划分；标准 checkpoint 路径由实验配置生成。 -->

## Configuration Groups

| Selection | Configuration directory | Purpose |
| --- | --- | --- |
| `model` | [configs/model/](../configs/model/) | Model handler, weights, tokenizer, templates |
| `trainer` | [configs/trainer/](../configs/trainer/) | Method and training arguments |
| `data` | [configs/data/](../configs/data/) | Dataset composition and preprocessing |
| `collator` | [configs/collator/](../configs/collator/) | Batch preparation |
| `eval` | [configs/eval/](../configs/eval/) | Evaluators and metric suites |
| `experiment` | [configs/experiment/](../configs/experiment/) | Complete experiment recipe |

<!-- 配置组分别管理模型、训练方法、数据、batch 整理、评估和完整实验。命令行选择组内 YAML，不需要重复传模型或方法名称。 -->

## Create an Experiment

Create `configs/experiment/unlearn/fiubench/my_npo.yaml` by inheriting an existing recipe:

<!-- 新实验优先继承已有流程，下例创建 FIUBench 的自定义 NPO 配置。 -->

```yaml
# @package _global_

defaults:
  - /experiment/unlearn/fiubench/default
  - override /model: llava-1.5-7b-hf
  - override /trainer: NPO
  - _self_

forget_split: forget_10
retain_split: retain_90
trainer:
  args:
    learning_rate: 5e-6
```

<!-- defaults 继承实验并覆盖组件，_self_ 将当前文件中的取值应用在前面的继承结果之后。包声明将配置放在全局，而不是 experiment 子节点。 -->

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/fiubench/my_npo
```

<!-- 创建配置后，通过实验路径运行，无需重复传入方法和超参数。 -->

Use command-line overrides for a single run, such as `trainer.args.learning_rate=5e-6`. Keep reusable settings in YAML. Method-specific recipes compose extra data and PEFT settings; see [MANU](../configs/experiment/unlearn/fiubench/manu.yaml) and [SMFA](../configs/experiment/unlearn/fiubench/smfa.yaml).

<!-- 单次调整可使用命令行覆盖，重复使用的设置写入 YAML；方法专用实验还可组合额外数据和 PEFT 设置。 -->

## Paths and Checkpoints

Data is prepared under `data/` by default. For another location, use the same path during setup and execution:

<!-- 数据默认准备到 data/；自定义目录时，准备与运行阶段应使用同一路径。 -->

```bash
python setup_data.py --fiubench --data-root /path/to/data
python src/train.py experiment=finetune/fiubench/default \
  paths.data_dir=/path/to/data
```

<!-- 准备阶段使用 data-root，实验阶段使用 paths.data_dir。 -->

Set `MMU_MODEL_ROOT=/path/to/models` to load locally downloaded model directories. Standard outputs live in `saves/finetune/`, `saves/unlearn/`, and `saves/eval/`; see [path defaults](../configs/paths/default.yaml). Model and method names follow the selected configurations.

<!-- 本地模型通过 MMU_MODEL_ROOT 指定父目录；标准输出位于 saves/ 下相应目录，具体路径见配置，名称跟随所选模型和方法。 -->

Evaluation experiments named `default`, `finetuned`, and `unlearned` select the original model, standard finetuned result, and unlearning result respectively. SafeEraser starts from the original model and provides `default` and `unlearned` evaluation recipes. For an external adapter, pass `checkpoint_path=/path/to/adapter`; for complete model weights, clear `model.adapter_path` and override `model.model_args.pretrained_model_name_or_path` instead.

<!-- default、finetuned、unlearned 分别评估原模型、微调结果、遗忘结果；SafeEraser 默认从原模型开始，提供原模型与遗忘评估。外部 adapter 用 checkpoint_path，完整模型则清空 adapter_path 并覆盖模型路径。 -->

## Launchers and Distributed Training

The benchmark launchers under [scripts/](../scripts/) select model, method, and split combinations and evaluate saved results. Training launchers default to two processes; set `NUM_PROCESSES=1` for one GPU. MANU and MIP-Editor are routed to one process automatically.

<!-- 启动脚本按模型、方法与划分组合运行并评估保存结果；训练默认两个进程，单卡设置 NUM_PROCESSES=1；MANU 与 MIP-Editor 自动使用单进程。 -->

```bash
DRY_RUN=1 bash scripts/fiubench_unlearn.sh
NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 bash scripts/fiubench_unlearn.sh
```

<!-- 第一条命令仅预览组合，第二条在 GPU 0 上单进程运行。 -->

## Preview a Configuration

Use `--cfg job` to inspect a composed configuration without loading models or running an experiment. Add `--resolve` to resolve interpolations:

<!-- 使用 cfg job 查看合成配置，不加载模型或执行实验；resolve 可展开插值。 -->

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/fiubench/default trainer=NPO --cfg job --resolve
python src/eval.py experiment=examples/mllmu_input_metrics --cfg job
python setup_data.py --fiubench --dry-run
```

<!-- 分别预览训练配置、评估配置和数据准备计划。 -->
