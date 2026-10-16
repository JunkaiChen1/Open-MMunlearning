# Contributing

Contributions of models, methods, benchmarks, metrics, bug fixes, and documentation are welcome. Start with the guide matching your change:

<!-- 欢迎贡献模型、方法、benchmark、指标、问题修复与文档；按变更类型阅读对应指南。 -->

| Guide | Covers |
| --- | --- |
| [Components](components.md) | Models, unlearning trainers, datasets, collators, registration |
| [Creating Evaluations](evaluation.md) | Metrics, dependencies, benchmark evaluators |
| [Configuring Experiments](experiments.md) | YAML composition, commands, checkpoints, launchers |

<!-- components.md 介绍组件接入，evaluation.md 介绍指标和 benchmark，experiments.md 介绍配置与运行。 -->

## Adding a Component

1. Implement the component using an existing interface and register its handler.
2. Add YAML configurations and a minimal experiment that selects the component.
3. Share an example command and, if applicable, a link to the related paper or implementation.

<!-- 新组件可按实现与注册、添加配置、提供示例命令的流程接入，有相关论文或实现时可附上链接。 -->

Existing components are useful starting points: metric files contain calculations, while evaluators handle benchmark data and inference. The linked guides show how to connect them through configurations.

<!-- 可从现有组件入手：metric 文件包含计算，evaluator 处理基准数据与推理，具体配置接入方式见上方指南。 -->

## Reporting and Submitting Changes

You can open an issue for a bug, a question, or an idea, or submit a pull request with an implementation. An example command and a short description help others understand the change. For bugs, the error message and environment details are useful when available.

<!-- 可以通过 issue 分享问题、疑问或想法，也可提交实现。附上示例命令和简短说明有助于理解变更，报告问题时可补充已有的报错和环境信息。 -->
