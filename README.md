<!-- Open-MMUnlearning 项目横幅：统一多模态大语言模型遗忘方法与评估。 -->

<p align="center">
  <img src="open-mmunlearning-v2.png?v=d2130dd4db" alt="Open-MMUnlearning: Unifying Methods and Evaluation for MLLM Unlearning" width="100%" />
  <br />
  <a href="https://arxiv.org/abs/2610.10358"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv Paper" /></a>
  <a href="https://github.com/JunkaiChen1/Open-MMunlearning/stargazers"><img src="https://img.shields.io/github/stars/JunkaiChen1/Open-MMunlearning?style=social" alt="GitHub Repo stars" /></a>
  <a href="https://github.com/JunkaiChen1/Open-MMunlearning"><img src="https://img.shields.io/github/repo-size/JunkaiChen1/Open-MMunlearning" alt="GitHub repo size" /></a>
  <a href="https://github.com/JunkaiChen1/Open-MMunlearning"><img src="https://img.shields.io/github/languages/top/JunkaiChen1/Open-MMunlearning" alt="GitHub top language" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT" /></a>
</p>

<!-- 顶部徽章依次为：论文、GitHub 星标、仓库大小、主要语言和 MIT 许可证。GitHub 地址采用论文公布的项目地址；Stars、大小和语言由 GitHub 线上仓库动态提供。 -->

## 📖 Overview

OpenMMUnlearning provides a unified, extensible framework for multimodal large language model (MLLM) unlearning, as described in [Open-MMUnlearning: Unifying Methods and Evaluation for MLLM Unlearning](https://arxiv.org/abs/2610.10358). It integrates multimodal data processing, target-model preparation, unlearning, and evaluation through shared interfaces and structured configurations. The repository includes **five unlearning benchmarks** covering privacy, safety, and copyright, **ten unlearning methods**, and **seven general multimodal evaluation suites**, together with benchmark-specific metrics, membership inference attacks, and adversarial input tests.

<!-- OpenMMUnlearning 为多模态大语言模型遗忘提供统一、可扩展的框架，相关设计见论文 Open-MMUnlearning。框架通过共享接口和结构化配置整合多模态数据处理、目标模型准备、模型遗忘与评估，当前仓库包含覆盖隐私、安全和版权的五个遗忘基准、十种遗忘方法、七套通用多模态评估，以及基准专用指标、成员推断攻击和对抗输入测试。 -->

## 🧩 Available Components

Following [Table 1 of the paper](https://arxiv.org/html/2610.10358v2#S3.T1), the table lists the models, unlearning methods, and datasets available in this repository.

<!-- 下表参考论文表 1，列出当前仓库支持的模型、遗忘方法和数据集。 -->

| Component | Category | Available variants |
| --- | --- | --- |
| Models | Multimodal backbones | LLaVA-1.5, LLaVA-1.6, Qwen2.5-VL, Gemma 3 |
| Methods | Unlearning | GradAscent, GradDiff, KL (`kl`), RMU, NPO, UNDIAL, MMUnlearner, MANU, MIP-Editor (`MIPEditor`), SMFA |
| Datasets | Privacy, safety, copyright | MLLMU-Bench, FIUBench, CLEAR, SafeEraser, CoVUBench |

<!-- 模型包括 LLaVA-1.5、LLaVA-1.6、Qwen2.5-VL 和 Gemma 3；十种遗忘方法统一列出；数据集覆盖隐私、安全与版权。 -->

## 📑 Contents

- [Quickstart](#-quickstart)
- [Preparing Data](#-preparing-data)
- [Running Experiments](#-running-experiments)
- [Running Evaluations](#-running-evaluations)
- [Extending the Framework](#-extending-the-framework)
- [Acknowledgements and Citation](#-acknowledgements-and-citation)
- [License](#-license)

<!-- 目录：快速开始、数据准备、实验运行、评估运行、框架扩展、致谢与引用、许可证。 -->

## ⚡ Quickstart

Use Python 3.11 and a CUDA-compatible PyTorch installation. Run the following commands from the repository root:

<!-- 使用 Python 3.11 和兼容当前 CUDA 环境的 PyTorch，在仓库根目录执行以下命令。 -->

```bash
conda create -n mmunlearning python=3.11
conda activate mmunlearning
python -m pip install -e .
```

<!-- 创建并激活环境，然后以可编辑模式安装本仓库。 -->

Dependencies are listed in [requirements.txt](requirements.txt). Install DeepSpeed or FlashAttention separately when required by the selected configuration.

<!-- 依赖版本见 requirements.txt；所选配置需要时，再单独安装 DeepSpeed 或 FlashAttention。 -->

## 📦 Preparing Data

Use [setup_data.py](setup_data.py) to download and prepare benchmark data:

<!-- 使用 setup_data.py 下载并准备基准数据。 -->

```bash
python setup_data.py --fiubench
python setup_data.py --mllmu --clear --covubench --safeeraser
python setup_data.py --eval-data
```

<!-- 可单独或组合选择遗忘基准，eval-data 用于准备通用多模态评估数据。 -->

Data sources and preparation settings are defined in [configs/data/download.yaml](configs/data/download.yaml). Use `--dry-run` to preview a plan and `--help` for available options. Prepared data is stored under `data/` by default.

<!-- 数据源与准备设置见 configs/data/download.yaml；dry-run 预览计划，help 查看选项，默认输出到 data/。 -->

## 🧪 Running Experiments

Experiments use Hydra YAML configurations to select datasets, models, methods, and hyperparameters.

<!-- 实验通过 Hydra YAML 配置选择数据集、模型、方法和超参数。 -->

### Finetuning

Finetune LLaVA-1.5-7B on FIUBench:

<!-- 在 FIUBench 上微调 LLaVA-1.5-7B。 -->

```bash
python src/train.py experiment=finetune/fiubench/default \
  model=llava-1.5-7b-hf
```

<!-- 加载 FIUBench 微调实验配置并指定模型。 -->

Replace `fiubench` with `mllmu`, `clear`, or `covubench` for other benchmarks. SafeEraser starts from the original pretrained model by default.

<!-- 替换基准名称可运行其他微调实验；SafeEraser 默认直接从预训练原模型开始遗忘。 -->

### Unlearning

Run NPO on FIUBench's 10% forget split after finetuning:

<!-- 完成微调后，在 FIUBench 的 10% 遗忘划分上运行 NPO。 -->

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/fiubench/default model=llava-1.5-7b-hf \
  trainer=NPO forget_split=forget_10 retain_split=retain_90
```

<!-- 使用 NPO 在 forget_10 与 retain_90 划分上执行遗忘。 -->

- `experiment` selects settings under [configs/experiment/](configs/experiment/).
- `model` selects a [model configuration](configs/model/).
- `trainer` selects the [unlearning method](configs/trainer/).
- `forget_split` and `retain_split` select training partitions.

<!-- experiment 选择实验配置，model 选择模型，trainer 选择遗忘方法，forget_split 和 retain_split 选择训练划分。 -->

MMUnlearner, MANU, MIP-Editor, and SMFA use dedicated experiments named `mmunlearner`, `manu`, `mip_editor`, and `smfa` under each benchmark:

<!-- MMUnlearner、MANU、MIP-Editor 和 SMFA 使用各基准下对应的专用实验配置。 -->

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/fiubench/manu model=llava-1.5-7b-hf
```

<!-- MANU 的方法与特殊设置由实验 YAML 自动选择。 -->

Models download from Hugging Face by default. For local models, set `MMU_MODEL_ROOT=/path/to/models`. Training outputs are saved under `saves/finetune/` and `saves/unlearn/`.

<!-- 模型默认从 Hugging Face 下载；使用本地模型时设置 MMU_MODEL_ROOT。微调与遗忘结果分别保存到 saves/finetune/ 和 saves/unlearn/。 -->

### Baseline Experiments

Use the launchers to run configured experiment combinations and their evaluations:

<!-- 使用启动脚本批量执行配置中的实验组合及对应评估。 -->

```bash
bash scripts/fiubench_finetune.sh
bash scripts/fiubench_unlearn.sh
```

<!-- 分别批量运行 FIUBench 微调与遗忘实验。 -->

Equivalent launchers are provided for the other benchmarks in [scripts/](scripts/). Edit the arrays to select models, methods, and splits; use `NUM_PROCESSES=1` for a single GPU or `DRY_RUN=1` to preview commands.

<!-- scripts/ 提供其他基准的启动脚本。修改数组选择模型、方法和划分，NUM_PROCESSES=1 使用单卡，DRY_RUN=1 仅预览命令。 -->

## 📊 Running Evaluations

Evaluate the saved FIUBench NPO checkpoint:

<!-- 评估已保存的 FIUBench NPO 遗忘结果。 -->

```bash
python src/eval.py experiment=eval/fiubench/unlearned \
  model=llava-1.5-7b-hf trainer=NPO ratio=10
```

<!-- 模型、方法和遗忘比例与训练保持一致，评估配置自动选择对应 checkpoint。 -->

Use `default` to evaluate the original model or `finetuned` for a finetuned model. Select metrics and benchmark settings in [configs/eval/](configs/eval/).

<!-- default 评估原模型，finetuned 评估微调模型；指标和基准设置见 configs/eval/。 -->

For general multimodal evaluation, select a suite and optionally provide an adapter:

<!-- 通用多模态评估通过实验名称选择基准，也可显式指定 adapter。 -->

```bash
python src/eval.py experiment=eval/pope/default \
  checkpoint_path=/path/to/unlearned-adapter
```

<!-- 示例在 POPE 上评估指定的遗忘 adapter；省略 checkpoint_path 则评估原模型。 -->

Replace `pope` with `mmvet`, `mmbench`, `vizwiz`, `scienceqa`, `gqa`, or `vqav2`. Attack examples are provided under [configs/experiment/examples/](configs/experiment/examples/) and [configs/experiment/eval/mllmu/](configs/experiment/eval/mllmu/).

<!-- 替换基准名称可选择其他通用评估；攻击示例见 examples/ 和 MLLMU 评估实验目录。 -->

## 🔧 Extending the Framework

To add models, methods, benchmarks, or metrics, implement a handler, register it, and select it through YAML. See the guides below:

<!-- 新增模型、方法、benchmark 或指标时，按“实现 handler、注册、通过 YAML 选择”的流程接入，具体步骤见以下指南。 -->

| Guide | Contents |
| --- | --- |
| [Contributing](docs/contributing.md) | Ways to contribute and share changes |
| [Components](docs/components.md) | Adding models, methods, datasets, and collators |
| [Creating Evaluations](docs/evaluation.md) | Adding metrics and benchmarks, inference dependencies |
| [Configuring Experiments](docs/experiments.md) | YAML composition, commands, checkpoints, and launchers |

<!-- 文档分别介绍贡献流程、模型与方法等组件接入、指标与 benchmark 接入，以及实验配置和运行。 -->

## 🙏 Acknowledgements and Citation

This project builds on [OpenUnlearning](https://github.com/locuslab/open-unlearning). We thank its authors for making their work open source.

<!-- 本项目基于 OpenUnlearning 构建，感谢作者开源相关工作。 -->

If you use OpenMMUnlearning in your research, please cite our paper:

<!-- 如果在研究中使用 OpenMMUnlearning，请引用我们的论文。 -->

```bibtex
@article{chen2026open,
  title={Open-MMUnlearning: Unifying Methods and Evaluation for MLLM Unlearning},
  author={Chen, Junkai and He, Yuhao and Wei, Qianshan and You, Junxiang and Shao, Jingwen and Lin, Junkai and Yue, Zhongkai and Ye, Xiaotian and Jiao, Zhengbo and Cheng, Jiali and others},
  journal={arXiv preprint arXiv:2610.10358},
  year={2026}
}
```

<!-- 上述 BibTeX 为 Open-MMUnlearning 论文的引用信息。 -->

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

<!-- 本项目采用 MIT 许可证，详见 LICENSE。 -->
