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

trainers_experiments=(
  "GradAscent unlearn/clear/default"
  "GradDiff unlearn/clear/default"
  "kl unlearn/clear/default"
  "NPO unlearn/clear/default"
  "RMU unlearn/clear/default"
  "UNDIAL unlearn/clear/default"
  "MMUnlearner unlearn/clear/mmunlearner"
  "MANU unlearn/clear/manu"
  "MIPEditor unlearn/clear/mip_editor"
  "SMFA unlearn/clear/smfa"
)

splits=(
  "forget_1 retain_99 1"
  "forget_5 retain_95 5"
  "forget_10 retain_90 10"
)

for model in "${models[@]}"; do
  for split in "${splits[@]}"; do
    read -r forget_split retain_split ratio <<< "$split"
    for trainer_experiment in "${trainers_experiments[@]}"; do
      read -r trainer experiment <<< "$trainer_experiment"
      launch=("$python_bin" -m accelerate.commands.launch
        --config_file configs/accelerate/ddp_config.yaml
        --num_processes "$num_processes" --main_process_port "$master_port")
      training_gpus="$train_gpus"
      if [[ "$num_processes" == "1" || "$trainer" == "MANU" || "$trainer" == "MIPEditor" ]]; then
        launch=("$python_bin")
      fi
      if [[ "$trainer" == "MANU" || "$trainer" == "MIPEditor" ]]; then
        training_gpus="$eval_gpu"
      fi

      CUDA_VISIBLE_DEVICES="$training_gpus" "${run[@]}" "${launch[@]}" \
        src/train.py --config-name=unlearn.yaml \
        experiment="$experiment" \
        model="$model" \
        trainer="$trainer" \
        forget_split="$forget_split" retain_split="$retain_split"

      CUDA_VISIBLE_DEVICES="$eval_gpu" "${run[@]}" "$python_bin" src/eval.py \
        experiment=eval/clear/unlearned \
        benchmark_name=clear model="$model" \
        trainer="$trainer" ratio="$ratio"

    done
  done
done
