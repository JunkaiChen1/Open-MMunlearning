# Native Method Integration

MIP-Editor and S-MLLMUn are integrated with the framework's Trainer contract.
Runtime code does not depend on external project checkouts.

## MIPEditor

The entry point is `src/trainer/unlearn/mip_editor.py`. The registered name is
`MIPEditor`, with configuration in `configs/trainer/MIPEditor.yaml`. During
training, the method collects the product of FFN `down_proj/fc2` input gradients
and activation importance on forget batches, selects the top-k input channels
per layer, and edits LoRA A. The language path scores answer tokens only, then
updates LoRA with retain loss and representation steering loss. The model must
contain LLaVA-, Qwen-, or Gemma-style FFN projection layers.

## SMFA

The entry point is `src/trainer/unlearn/smfa.py`. The registered name is `SMFA`,
with configuration in `configs/trainer/SMFA.yaml`. One Trainer maintains three
LoRA adapters: `MFA_multi`, `MFA_text`, and `RA`. The two MFA adapters learn IDK
forget targets and real retain answers on multimodal and text-only views; RA
learns both retain views. The save directory contains the three adapters and
`unlearning_artifact.json`. Evaluation applies sculpting with `multi_k/text_k`
to the fine-tuned base model, avoiding a roughly 13 GiB full-model write for
each trial.

The original S-MLLMUn LLaVA-OneVision baseline used `float16`, batch size 2,
learning rate `1e-4`, 5 epochs, LoRA `r=16/alpha=16/dropout=0.05`, and fixed
sculpt values `multi_k=5` and `text_k=10`. The current implementation also uses
random IDK targets and minimizes cross-entropy, but combines six data branches
into one framework optimizer step. Its scheduling therefore differs from the
four independent dataloaders in the original repository. Formal LLaVA-1.5-7B
runs default to `bfloat16`.

## Running

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/mllmubench/mip_editor_llava7b \
  model=llava-1.5-7b-hf trainer=MIPEditor peft=lora \
  forget_split=forget_10 retain_split=retain_90 task_name=MIP_NATIVE

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/mllmubench/smfa_llava7b \
  model=llava-1.5-7b-hf trainer=SMFA peft=lora \
  forget_split=forget_10 retain_split=retain_90 task_name=SMFA_NATIVE
```

The unified scheduler `scripts/run_unlearning_search.py` also supports both
methods:

```text
MIP_EDITOR: lr={1e-4,5e-4,1e-3} x path_topk={3,5,10} x steering={0.5,1,2} = 27
SMFA:       lr={5e-5,1e-4,2e-4} x multi_k={2.5,5,10} = 9
```

The default 114-trial search keeps the original eight methods. Add these two
native methods explicitly with `--methods ...,MIP_EDITOR,SMFA`.
