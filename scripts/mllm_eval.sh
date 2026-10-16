#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-python}"
eval_gpu="${EVAL_GPU:-0}"
run=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  run=(echo)
fi

models=(
  "llava-1.5-7b-hf"
)

benchmarks=(
  "pope"
  "mmvet"
  "mmbench"
  "vizwiz"
  "scienceqa"
  "gqa"
  "vqav2"
)

for model in "${models[@]}"; do
  for benchmark in "${benchmarks[@]}"; do
    CUDA_VISIBLE_DEVICES="$eval_gpu" "${run[@]}" "$python_bin" src/eval.py \
      experiment=eval/mllm_eval/default eval="$benchmark" \
      model="$model"
  done
done
