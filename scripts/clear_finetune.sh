#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-python}"
train_gpus="${CUDA_VISIBLE_DEVICES:-0,1}"
eval_gpu="${EVAL_GPU:-0}"
num_processes="${NUM_PROCESSES:-2}"
master_port="${MASTER_PORT:-29500}"
run=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  run=(echo)
fi

models=(
  "llava-1.5-7b-hf"
)

eval_ratios=(1 5 10)

launch=("$python_bin" -m accelerate.commands.launch
  --config_file configs/accelerate/ddp_config.yaml
  --num_processes "$num_processes" --main_process_port "$master_port")
if [[ "$num_processes" == "1" ]]; then
  launch=("$python_bin")
fi

for model in "${models[@]}"; do
  CUDA_VISIBLE_DEVICES="$train_gpus" "${run[@]}" "${launch[@]}" \
    src/train.py experiment=finetune/clear/default \
    model="$model"

  for ratio in "${eval_ratios[@]}"; do
    CUDA_VISIBLE_DEVICES="$eval_gpu" "${run[@]}" "$python_bin" src/eval.py \
      experiment=eval/clear/finetuned \
      benchmark_name=clear model="$model" \
      ratio="$ratio"
  done
done
